


def train_tag_embs(checkpoint_path=None):
	#load stuff
	data_loader = torch.load(tag_data_path)
	n_tag, n_ques = pickle_load(data_meta_path)
	tag_names = pickle_load(tag_names_path)

	#init model
	model = skip_gram(n_tag, n_ques, emb_dim)

	#train & extract
	train_embs(model, data_loader, seed_val, checkpoint_path, main_model_path)
	extract_embs(main_model_path, ques_path, tag_list, emb_csv_path, emb_meta_path, True)



def train_item_embs(tag_emb_path=None, check_point_path=None):
	#load stuff
	data_loader = torch.load(main_data_path)
	n_tag, n_ques = pickle_load(data_meta_path)
	tag_names = pickle_load(tag_names_path)

	#init model
	if tag_emb_path:
		model = skip_gram_split_2(n_tag, n_ques, emb_dim)
		init_tag_embs = read_embs(tag_emb_path)
		sd = model.state_dict()
		sd['tag_embs.weight'] = init_tag_embs
		model.state_dict = sd
	else:
		model = skip_gram(n_tag+n_ques, n_tag, emb_dim)

	#train & extract
	train_embs(model, data_loader, seed_val, checkpoint_path, main_model_path)
	extract_embs(main_model_path, ques_path, tag_list, emb_csv_path, emb_meta_path, False)




def train_embs(model, data_loader, seed_val, checkpoint_path, dump_path):
	t0_overall  =  time()

	#move model to cuda
	device = torch.device('cuda')
	model.cuda()

	#prep optimizer & criterion
	optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-8)
	criterion = nn.BCELoss()

	#load checkpoint if exists
	if checkpoint_path:
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint['model_state'])
		optimizer.load_state_dict(checkpoint['optimizer_state'])
		last_epoch = checkpoint['epoch']
	else:
		last_epoch = 0

	#seed torch
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)

	#train loop
	train_losses = []
	for epoch_i in range(last_epoch, last_epoch + epochs):
		print('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, last_epoch + epochs))

		train_loss = 0.0
		model.train()
		t0_epoch = time()

		#train epoch pass
		for step, batch in enumerate(data_loader):
			b_input_ids = batch[0].to(device)
			b_labels = batch[1].to(device)
			model.zero_grad()
			outputs = model(b_input_ids)
			loss = criterion(outputs, b_labels)
			train_loss += loss.item()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #research this
			optimizer.step()
			scheduler.step()

		avg_train_loss = train_loss / len(data_loader)
		train_losses.append(avg_train_loss)
		print('  Average training loss: {0:.6f}'.format(avg_train_loss))
		print('  Training epcoh took: {0:.2f}'.format(time() - t0_epoch))
		print('  Cuda memory usage: {0:.2f}GB'.format(torch.cuda.memory_allocated(0)/1024**3))

	print('\nTrain completed; total time: {0:.2f}'.format(time() - t0_overall))

	#save checkpoint
	model.eval()
	checkpoint = {
		'epoch': last_epoch + epochs,
		'model_state': model.state_dict(),
		'optimizer_state':optimizer.state_dict()
	}
	torch.save(checkpoint, dump_path)
	plot_loss(train_losses)


def extract_embs(checkpoint_path, ques_path, tag_list, emb_csv_path, emb_meta_path, tag_only):
	#load embs
	state_dict = torch.load(model_path)['model_state']
	if tag_only:
		embs = state_dict['embeddings.weight'].detach().cpu().numpy()
	else:
		tag_embs = state_dict['tag_embs.weight'].detach().cpu().numpy()
		ques_embs = state_dict['ques_embs.weight'].detach().cpu().numpy()
		embs = np.concatenate((tag_embs, ques_embs), axis=0)

	#get items
	ques_df = pickle_load(ques_path)
	tags = tag_list
	questions = list(ques_df['title'])
	ques_tags = list(ques_df['tags'])

	#write embs
	with open(emb_path, 'w') as f:
		for emb in embs:
			line = '\t'.join([str(x) for x in emb]) + '\n'
			f.write(line)

	#write meta
	with open(emb_meta_path, 'w', encoding='utf-8') as f:
		f.write('name\ttype\ttags\n')
		for tag in tags:
			f.write('{}\t{}\t{}\n'.format(tag, 'tag', tag))
		if not tag_only:
			for ques, ques_tag in zip(questions, ques_tags):
				f.write('{}\t{}\t{}\n'.format(ques, 'question', ques_tag))

	print('dumped embs to files')











































