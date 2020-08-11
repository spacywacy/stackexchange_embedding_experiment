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
from sklearn.model_selection import train_test_split


#names
data_name = 'mixed_400_page'
model_name = 'mixed_400_page'

#data path
ques_path = 'data/{}_ques_df.pickle'.format(data_name)
tag_site_path = 'data/{}_tag_site.pickle'.format(data_name)

#meta path
tag_emb_path = 'embs/{}_tag_embs.tsv'.format(model_name)
main_emb_path = 'embs/{}_main_embs.tsv'.format(model_name)

#train params
train_batch_size = 8




def main():
	prep_tag_data()
	#test_split()


def pickle_dump(fname, obj_):
	with open(fname, 'wb') as f:
		pickle.dump(obj_, f)

def pickle_load(fname):
	with open(fname, 'rb') as f:
		obj_ = pickle.load(f)
	return obj_

def read_embs(emb_path):
	embs = []
	with open(emb_path, 'r') as f:
		for line in f:
			row = [float(x) for x in line[:-1].split('\t')]
			embs.append(row)

	embs = np.array(embs)
	return embs


def prep_tag_data():
	#load stuff
	tag_embs = read_embs(tag_emb_path)
	tag_sites = pickle_load(tag_site_path)

	#prep labels
	label_dict = {
		'stackoverflow': 0,
		'crossvalidated': 1
	}
	tag_sites = np.array([label_dict[x] for x in tag_sites])

	#train test split
	xtrain, xtest, ytrain, ytest = train_test_split(tag_embs, tag_sites)
	
	#cast to tensors
	xtrain = torch.tensor(xtrain, dtype=torch.float)
	xtest = torch.tensor(xtest, dtype=torch.float)
	ytrain = torch.tensor(ytrain, dtype=torch.float)
	ytrain.unsqueeze(0)
	ytrain = ytrain.reshape(-1, 1)
	ytest = torch.tensor(ytest, dtype=torch.float)
	ytest.unsqueeze(0)
	ytest = ytrain.reshape(-1, 1)

	#build train data loader
	trainset = TensorDataset(xtrain, ytrain)
	train_sampler = RandomSampler(trainset)
	train_loader = DataLoader(trainset, sampler=train_sampler, batch_size=train_batch_size)

	return train_loader, xtest, ytest

def prep_ques_data():
	pass

def prep_main_data():
	pass


def train(model, data_loader, epochs, seed_val, model_path):
	#this might need some revision to do cls

	t00 = time()

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
		print('  Training epcoh took: {0:.2f}'.format(time() - t0))
		print('  Cuda memory usage: {0:.2f}GB'.format(torch.cuda.memory_allocated(0)/1024**3))

	print('train completed; total time: {0:.2f}'.format(time() - t00))
	#torch.save(model.state_dict(), model_path)
	torch.save(model.state_dict, model_path)
	plot_loss(train_losses)




















if __name__ == '__main__':
	main()








