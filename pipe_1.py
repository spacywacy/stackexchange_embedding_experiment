import os
import pickle
import dill
import requests
import json
from urllib.parse import quote
import numpy as np
import pandas as pd
from time import time
from time import sleep
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup


#enable the following:
	#prep reversed data set to train tag embs
	#freeze tag embs for main model
	#use ques & tag meta data and trained embs to classify sites



#names
data_name = 'mixed_400_page'
model_name = 'mixed_400_page_test_cp'

#build data params
api_key = 'EfszLp6dWEhyCHQ8fxGpWA(('
sites = ['crossvalidated', 'stackoverflow']
n_pages = 200 #n_pages for each site
page_size = 20
ques_path = 'data/{}_ques_df.pickle'.format(data_name)
tag_data_path = 'data/{}_tag_dataloader.pickle'.format(data_name)
main_data_path = 'data/{}_main_dataloader.pickle'.format(data_name)
tag_names_path = 'data/{}_tag_names.pickle'.format(data_name)
tag_site_path = 'data/{}_tag_site.pickle'.format(data_name)
data_meta_path = 'data/{}_data_meta.pickle'.format(data_name)

#train params
batch_size = 1
tag_model_batch_size = 1
main_model_batch_size = 1
tag_model_epochs = 5
main_model_epochs = 5
seed_val = 142
lr = 5e-4

#model params
emb_dim = 30
tag_model_path = 'models/{}_tag_checkpoint.pt'.format(model_name)
main_model_path = 'models/{}_main_checkpoint.pt'.format(model_name)
#tag_check_point_path = 'models/{}_tag_check_point'.format(model_name)
#main_check_point_path = 'models/{}_main_check_point'.format(model_name)

#emb params
tag_emb_path = 'embs/{}_tag_embs.tsv'.format(model_name)
tag_emb_meta_path = 'embs/{}_tag_embs_meta.tsv'.format(model_name)
main_emb_path = 'embs/{}_main_embs.tsv'.format(model_name)
main_emb_meta_path = 'embs/{}_main_embs_meta.tsv'.format(model_name)

#device
device = torch.device('cuda')




def main():
	#get data from api
	#create data loader for main model
	#[optional]create data loader for tag model
	#[optional]train tag model
	#[optional]extract tag embs
	#[optional]init main model with tag embs
	#[optional]freeze tag embs in main model
	#train main model
	#extract ques & tag embs from main model

	#data_from_api()
	#process_raw()
	#train_tag_embs()
	#train_tag_embs(checkpoint_path=tag_model_path)
	#train_item_embs(tag_emb_path=tag_emb_path)
	train_item_embs(tag_emb_path=tag_emb_path, checkpoint_path=main_model_path)


########################################
#helper functions

def pickle_dump(fname, obj_):
	with open(fname, 'wb') as f:
		pickle.dump(obj_, f)

def pickle_load(fname):
	with open(fname, 'rb') as f:
		obj_ = pickle.load(f)
	return obj_

def call_api(url, params, raw=False):
	res = requests.get(url, params=params)
	url_obj = res.content.decode('utf-8')
	if raw:
		return url_obj
	else:
		try:
			return json.loads(url_obj)['items']
		except:
			print(url_obj)

def id2vec(i, n_items):
	vec = [0]*n_items
	vec[i] = 1
	return vec

def plot_loss(loss):
	xs = list(range(len(loss)))
	plt.plot(xs, loss)
	plt.show()

def read_embs(emb_path):
	embs = []
	with open(emb_path, 'r') as f:
		for line in f:
			row = [float(x) for x in line[:-1].split('\t')]
			embs.append(row)

	embs = torch.tensor(embs, dtype=torch.float)
	return embs



########################################



########################################
#build data

def data_from_api():
	ques_df = []

	for site in sites:
		for i in range(n_pages):
			#api call
			print('getting page:', i)
			url = 'https://api.stackexchange.com/2.2/questions'
			params = {
				'order':'desc',
				#'sort':'activity',
				'sort':'votes',
				'site':site,
				'key':api_key,
				'page':i+1,
				'pagesize':page_size
			}
			json_items = call_api(url, params)

			#build questions dataframe
			for id_, item in enumerate(json_items):
				title = item.get('title')
				url = item.get('link')
				tags = ','.join([str(x) for x in item.get('tags')])
				ques_df.append([id_, title, url, tags, site])

			sleep(1)
	
	#build ques_df
	ques_df = pd.DataFrame(ques_df, columns=['id', 'title', 'url', 'tags', 'site'])
	print('ques_df:')
	print(ques_df, ques_df.shape)

	#dump files
	pickle_dump(ques_path, ques_df)
	print('saved raw ques df')

def process_raw():
	###what's happening here
	#<tag model>
	#tag model in: tag_ids
	#tag model out: ques_one_hot_sums

	#<main model>
	#main model in:
		#tag_ids
		#ques_ids + n_tag
	#main model out:
		#tag_one_hots
		#tag_one_hot_sums

	#<tag2site model>
	#in: tag_id, tag_name
	#out: tag_site_list

	#<ques2site model>
	#use ques_df


	#load raw
	ques_df = pickle_load(ques_path)

	#get tag names
	tag_names = []
	for tags_str in ques_df['tags']:
		tag_names += tags_str.split(',')
	tag_names = list(set(tag_names))
	tag_names.sort()
	n_tag = len(tag_names)


	#get tag one hot
	tag_ids = []
	tag_one_hots = []
	tag_dict = {} #tag_name: (i, tag_one_hot)

	for i, tag_name in enumerate(tag_names):
		tag_ids.append(i)
		tag_one_hot = id2vec(i, n_tag)
		tag_one_hots.append(tag_one_hot)
		tag_dict[tag_name] = (i, tag_one_hot)


	#get tag one hot sums
	def tags2vec(tags):
		'''subroutine: get tag vec sum from tag names'''
		tags_list = tags.split(',')
		tag_vecs = np.array([tag_dict[x][1] for x in tags_list])
		return np.sum(tag_vecs, axis=0)

	ques_ids = list(np.array(ques_df.index) + n_tag)
	tag_vec_sums = list(ques_df['tags'].apply(tags2vec))


	#get ques one hot & one hot sums
	n_ques = len(ques_df)
	tag_target_ques = {} #tag_name: ques_one_hot_sum
	tag_site_dict = {} #tag_name: site
	tag_sites = []
	ques_one_hot_sums = []

	for i, row in ques_df.iterrows():
		ques_one_hot = np.array(id2vec(i, n_ques))
		for tag_name in row['tags'].split(','):
			tag_target_ques[tag_name] = tag_target_ques.get(tag_name, np.array([0]*n_ques)) + ques_one_hot
			tag_site_dict[tag_name] = row['site']

	for tag_name in tag_names:
		ques_one_hot_sums.append(tag_target_ques[tag_name])
		tag_sites.append(tag_site_dict[tag_name])


	#test prints
	#print('tag_ids')
	#print(tag_ids)
	#print()
	#print('ques_one_hot_sums')
	#print(ques_one_hot_sums[3])
	#test_tag = tag_names[3]
	#picked_ques = []
	#for i, row in ques_df.iterrows():
		#if test_tag in row['tags'].split(','):
			#picked_ques.append(i)
	#print(picked_ques)
	#print(ques_one_hot_sums[3][6835])
	#print(ques_one_hot_sums[3][7248])
	#print(ques_one_hot_sums[3][7421])
	#for a, b in zip(tag_names[2000:2010], tag_sites[2000:2010]):
		#print(a, b)
	#return


	
	#build tag model data loader
	tag_model_x = torch.tensor(tag_ids, dtype=torch.long)
	tag_model_x.unsqueeze(0)
	tag_model_x = tag_model_x.reshape(-1,1)
	tag_model_y = torch.tensor(ques_one_hot_sums, dtype=torch.float)
	tag_model_train = TensorDataset(tag_model_x, tag_model_y)
	tag_model_sampler = RandomSampler(tag_model_train)
	tag_model_data_loader = DataLoader(tag_model_train, sampler=tag_model_sampler, batch_size=tag_model_batch_size)


	#build main model data loader
	main_model_x = torch.tensor(tag_ids + ques_ids, dtype=torch.long)
	main_model_x.unsqueeze(0)
	main_model_x = main_model_x.reshape(-1,1)
	main_model_y = torch.tensor(tag_one_hots + tag_vec_sums, dtype=torch.float)
	main_model_train = TensorDataset(main_model_x, main_model_y)
	main_model_sampler = RandomSampler(main_model_train)
	main_model_data_loader = DataLoader(main_model_train, sampler=main_model_sampler, batch_size=main_model_batch_size)

	#need separated data loaders for tag and ques


	#dump results
	torch.save(tag_model_data_loader, tag_data_path)
	torch.save(main_model_data_loader, main_data_path)
	pickle_dump(data_meta_path, (n_tag, n_ques))
	pickle_dump(tag_names_path, tag_names)
	pickle_dump(tag_site_path, tag_sites)
	print('dumped files')


########################################



########################################
#train embs

def train_tag_embs(checkpoint_path=None):
	#load stuff
	data_loader = torch.load(tag_data_path)
	n_tag, n_ques = pickle_load(data_meta_path)
	tag_names = pickle_load(tag_names_path)

	#init model
	model = skip_gram(n_tag, n_ques, emb_dim)

	#train & extract
	train_embs(model, data_loader, seed_val, tag_model_epochs, checkpoint_path, tag_model_path)
	extract_embs(tag_model_path, ques_path, tag_names, tag_emb_path, tag_emb_meta_path, True)


def train_item_embs(tag_emb_path=None, checkpoint_path=None):
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
		#model.state_dict = sd
		model.load_state_dict(sd)
	else:
		model = skip_gram(n_tag+n_ques, n_tag, emb_dim)

	#train & extract
	train_embs(model, data_loader, seed_val, main_model_epochs, checkpoint_path, main_model_path)
	extract_embs(main_model_path, ques_path, tag_names, main_emb_path, main_emb_meta_path, False)


def train_embs(model, data_loader, seed_val, epochs, checkpoint_path, dump_path):
	t0_overall = time()

	#move model to cuda
	device = torch.device('cuda')
	model.cuda()

	#prep optimizer & criterion
	optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-8)
	criterion = nn.BCELoss()

	#load checkpoint if exists
	if checkpoint_path:
		print('training from existing checkpoint')
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint['model_state'])
		optimizer.load_state_dict(checkpoint['optimizer_state'])
		last_epoch = checkpoint['epoch']
	else:
		print('training from fresh model')
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


class skip_gram(nn.Module):

	def __init__(self, n_items, n_output, emb_dim):
		super().__init__()
		self.embeddings = nn.Embedding(n_items, emb_dim)
		self.fc = nn.Linear(emb_dim, n_output)
		self.softmax = nn.Softmax(-1)

	def forward(self, item_ids):
		z = self.embeddings(item_ids)
		z = self.fc(z)
		z = self.softmax(z)
		return z

class skip_gram_split(nn.Module):

	def __init__(self, n_tag, n_ques, emb_dim):
		super().__init__()
		self.tag_embs = nn.Embedding(n_tag, emb_dim)
		self.tag_embs.weight.require_grad = False
		self.ques_embs = nn.Embedding(n_ques, emb_dim)
		self.fc = nn.Linear(emb_dim, n_tag)
		self.softmax = nn.Softmax(-1)
		self.n_tag = n_tag

	def forward(self, tag_ids, ques_ids):
		tag_z = self.tag_embs(tag_ids)
		ques_z = self.ques_embs(ques_ids-self.n_tag)
		z = torch.cat((tag_z, ques_z), dim=0)
		z = self.fc(z)
		z = self.softmax(z)
		return z

class skip_gram_split_2(nn.Module):

	def __init__(self, n_tag, n_ques, emb_dim):
		super().__init__()
		self.tag_embs = nn.Embedding(n_tag, emb_dim)
		self.tag_embs.weight.require_grad = False
		self.ques_embs = nn.Embedding(n_ques, emb_dim)
		self.fc = nn.Linear(emb_dim, n_tag)
		self.softmax = nn.Softmax(-1)
		self.n_tag = n_tag

	def forward(self, item_ids):
		#split item_ids into tag & ques ids
		tag_mask = item_ids < self.n_tag
		ques_mask = item_ids >= self.n_tag
		tag_ids = item_ids[tag_mask]
		ques_ids = item_ids[ques_mask]

		#forward pass
		tag_z = self.tag_embs(tag_ids)
		ques_z = self.ques_embs(ques_ids-self.n_tag)
		z = torch.cat((tag_z, ques_z), dim=0)
		z = self.fc(z)
		z = self.softmax(z)

		return z


########################################



########################################
#extract emb

def extract_embs(checkpoint_path, ques_path, tag_list, emb_csv_path, emb_meta_path, tag_only):
	#load embs
	state_dict = torch.load(checkpoint_path)['model_state']
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
	with open(emb_csv_path, 'w') as f:
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





########################################





















if __name__ == '__main__':
	main()






