

def train_item_embs(tag_emb_path=None, model_path=None, check_point_path=None):
	#load stuff
	data_loader = torch.load(main_data_path)
	n_tag, n_ques = pickle_load(data_meta_path)
	tag_names = pickle_load(tag_names_path)


	#train & extract embs
	if tag_emb_path:
		#init model & tag embs
		init_tag_embs = read_embs(tag_emb_path)
		model = skip_gram_split_2(n_tag, n_ques, emb_dim)
		model.cuda()

		#load model state if exists
		if model_path:
			print('loading existing model states')
			model.load_state_dict(torch.load(model_path))
		else:
			print('train new model')
			sd = model.state_dict()
			sd['tag_embs.weight'] = init_tag_embs
			model.state_dict = sd

		#load checkpoint if exists
		if check_point_path:
			print('loading train checkpoint')
			checkpoint = torch.load(check_point_path, pickle_module=dill)
			#optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-8)
			optimizer = optim.Adam(model.parameters())
			optimizer.load_state_dict(checkpoint['optimizer_state'])
			#scheduler = checkpoint['scheduler']
			total_steps = len(data_loader) * main_model_epochs
			scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
			last_epoch = checkpoint['epoch']
		else:
			print('train new model')
			optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-8)
			total_steps = len(data_loader) * main_model_epochs
			scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
			last_epoch = 0

		#train & extract embs
		train_embs(model, data_loader, main_model_epochs, seed_val, main_model_path, main_check_point_path, optimizer, scheduler, last_epoch)
		extract_embs(main_model_path, ques_path, tag_names, main_emb_path, main_emb_meta_path, False)

	else:
		model = skip_gram(n_tag+n_ques, n_tag, emb_dim)
		model.cuda()

		#laod model state if exists
		if model_path:
			print('loaded existing model state')
			model.load_state_dict(torch.load(model_path))

		#load checkpoint if exists
		if check_point_path:
			print('loading train checkpoint')
			checkpoint = torch.load(check_point_path)
			optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-8)
			optimizer.load_state_dict(checkpoint['optimizer_state'])
			scheduler = checkpoint['scheduler']
			last_epoch = checkpoint['epoch']
		else:
			print('train new model')
			optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-8)
			total_steps = len(data_loader) * main_model_epochs
			scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
			last_epoch = 0


		train_embs(model, data_loader, main_model_epochs, seed_val, main_model_path, main_check_point_path, optimizer, scheduler, last_epoch)
		extract_embs(main_model_path, ques_path, tag_names, main_emb_path, main_emb_meta_path, False)












































