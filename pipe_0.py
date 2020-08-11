import os
import pickle
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


#<main process>
#build data
	#get top tags df
		#id, tag, n_ques, emb
	#get question df & tag group
		#ques_df: id, title, url, raw_tags, top_tags
		#tag group: nested list of top tags in each question
	#get pairs
		#input: tag_ids + ques_ids
		#output: one hot vecs of top tags
#train
	#base model:
		#treat tag & ques as same thing
		#not perserving relationships between tags
	#initialize embs with tag embs
	#use & freeze tag embs
#extract emb
	#load state dict & extract embs
	#write tsv files

#<tag emb>
#use top_tags from ques_df to get tag pairs
#train tag embs using tag pairs
#insert tag embs before main model train


#<next steps>
#scale up the dataset
#build persistent db

#names
data_name = 'cv_200_page'
model_name = 'cv_200_page'

#build data params
api_key = 'EfszLp6dWEhyCHQ8fxGpWA(('
#site = 'stackoverflow'
site = 'crossvalidated'
n_pages = 200
page_size = 20
ques_path = 'data/{}_ques_df.pickle'.format(data_name)
tag_groups_path = 'data/{}_tag_groups.pickle'.format(data_name)

#train params
batch_size = 1
epochs = 200
seed_val = 142
lr = 5e-4

#model params
emb_dim = 30
model_path = 'models/{}_ques_embs.pt'.format(model_name)

#emb params
emb_path = 'embs/{}_embs.tsv'.format(model_name)
emb_meta_path = 'embs/{}_embs_meta.tsv'.format(model_name)





def main():
	get_ques_tag(api_key, site, n_pages, page_size, ques_path, tag_groups_path)
	#data_loader, n_tag, n_ques, tag_list = build_data(api_key, site, n_pages, page_size, ques_path, tag_groups_path)
	#n_items = n_tag + n_ques
	#model = skip_gram(n_items, n_tag, emb_dim)
	#train(model, data_loader, epochs, seed_val, model_path)
	#extract_embs(model_path, ques_path, tag_list, emb_path, emb_meta_path)



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

########################################




########################################
#build data

def get_ques_tag(api_key, site, n_pages, page_size, ques_path, tag_groups_path):
	'''
	query api to get questions and their tags
	dump questions to file as df
	dump tag groups to file as nexted list
	'''

	tag_groups = []
	ques_df = []

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

		#get tag groups
		tag_groups += [x.get('tags') for x in json_items]

		#build questions dataframe
		for id_, item in enumerate(json_items):
			title = item.get('title')
			url = item.get('link')
			tags = ','.join([str(x) for x in item.get('tags')])
			ques_df.append([id_, title, url, tags])

		sleep(1)
	
	#summary
	ques_df = pd.DataFrame(ques_df, columns=['id', 'title', 'url', 'tags'])
	print('tag groups len:', len(tag_groups))
	print('ques_df:')
	print(ques_df, ques_df.shape)

	#dump files
	pickle_dump(ques_path, ques_df)
	pickle_dump(tag_groups_path, tag_groups)
	print('saved files')


def build_tag(tag_groups_path):
	'''
	in: nested list of tag groups
	return:
		[list] tag_indices
		[nested list] tag_vecs
		[dict] tag_name: (tag_index, tag_vec)
	'''
	
	#load tag groups
	groups = pickle_load(tag_groups_path)

	#create tag list
	tag_list = []
	for group in groups:
		tag_list += group
	tag_list = list(set(tag_list))
	tag_list.sort()
	n_tags = len(tag_list)

	#crete tag vecs
	tag_inds = []
	tag_vecs = []
	tag_dict = {}

	for i, tag in enumerate(tag_list):
		tag_vec = id2vec(i, n_tags)
		tag_inds.append(i)
		tag_vecs.append(tag_vec)
		tag_dict[tag] = (i, tag_vec)

	#for a,b in zip(tag_inds, tag_vecs):
		#print(a,b)
	#for k, v in tag_dict.items():
		#print(k, v)

	return tag_inds, tag_vecs, tag_dict, tag_list


def build_ques(ques_path, tag_dict):
	'''
	in:
		ques df
		tag_name: (tag_index, tag_vec)
	return: 
		[list] ques_indices
		[neste list] tag_vec_sum
	'''
	
	#load ques_df
	ques_df = pickle_load(ques_path)

	#subroutine: get tag vec sum from tag names
	def tags2vec(tags):
		tags_list = tags.split(',')
		tag_vecs = np.array([tag_dict[x][1] for x in tags_list])
		return np.sum(tag_vecs, axis=0)

	n_tags = len(tag_dict)
	ques_inds = list(np.array(ques_df.index) + n_tags)
	tag_vec_sums = list(ques_df['tags'].apply(tags2vec))

	#print(ques_inds)
	#print(tag_vec_sums)
	#print(type(tag_vec_sums))

	return ques_inds, tag_vec_sums


def build_data(api_key, site, n_pages, page_size, ques_path, tag_groups_path):
	'''
	in:
		[nested list] tag_index: tag_vec
		[nested list] ques_index: tag_vec_sum
	dump to file:
		[np array] tag + ques index: tag_vec(_sum)
	'''

	#transform raw pickled data
	tag_inds, tag_vecs, tag_dict, tag_list = build_tag(tag_groups_path)
	ques_inds, tag_vec_sums = build_ques(ques_path, tag_dict)

	#get n_ques & n_tag
	n_tag = len(tag_inds)
	n_ques = len(ques_inds)

	#build tensors
	in_ids = torch.tensor(tag_inds + ques_inds, dtype=torch.long)
	in_ids = in_ids.unsqueeze(0)
	in_ids = in_ids.reshape(-1, 1)
	target_vecs = torch.tensor(tag_vecs + tag_vec_sums, dtype=torch.float)

	#prep data loader
	train_data = TensorDataset(in_ids, target_vecs)
	train_sampler = RandomSampler(train_data)
	data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

	return data_loader, n_tag, n_ques, tag_list


########################################




########################################
#train emb
def train(model, data_loader, epochs, seed_val, model_path):
	#prep model & optimizer
	device = torch.device('cuda')
	model.cuda()
	optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-8)
	criterion = nn.BCELoss()
	total_steps = len(data_loader) * epochs
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

	#seed torch
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)

	#train
	train_losses = []
	for epoch_i in range(0, epochs):
		print('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
		print('Training...')

		train_loss = 0
		model.train()
		t0 = time()

		#train epoch pass
		for step, batch in enumerate(data_loader):
			if step % 100 == 0 and not step == 0:
				elapsed = time() - t0
				#print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(data_loader), elapsed))

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
		print('\n  Average training loss: {0:.6f}'.format(avg_train_loss))
		#print('  Training epcoh took: {0:.2f}'.format(time() - t0))

	print('train completed ')
	torch.save(model.state_dict(), model_path)
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


########################################



########################################
#extract emb

def extract_embs(model_path, ques_path, tag_list, emb_path, emb_meta_path):
	#load embs
	state_dict = torch.load(model_path)
	embs = state_dict['embeddings.weight'].detach().cpu().numpy()
	#print(embs)

	#get items
	ques_df = pickle_load(ques_path)
	tags = tag_list
	questions = list(ques_df['title'])
	ques_tags = list(ques_df['tags'])

	#for item, emb in zip(items, embs):
		#print(item, emb)

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
		for ques, ques_tag in zip(questions, ques_tags):
			f.write('{}\t{}\t{}\n'.format(ques, 'question', ques_tag))

	print('dumped embs to files')
	







########################################



if __name__ == '__main__':
	main()


