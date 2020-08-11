import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


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
		#z = self.embeddings(item_ids)
		tag_z = self.tag_embs(tag_ids)
		#print(ques_ids)
		ques_z = self.ques_embs(ques_ids-self.n_tag)
		#print(tag_z)
		#print(ques_z)
		z = torch.cat((tag_z, ques_z), dim=0)
		#print(z)
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


#3 ways to train item embs while freezing tag embs
	#have the forward() split incoming item_ids into tag_ids & ques_ids
	#create two separated data loaders for tag & ques
	#build a custom dataloader that loads a ques & its tags at a time



if __name__ == '__main__':
	#model = skip_gram_split(5, 15, 5)
	model = skip_gram_split_2(5, 15, 5)
	#for p in model.parameters():
		#print(p)

	'''
	sd = model.state_dict()
	#print(sd['embeddings.weight'])
	new_embs = [[1 for x in range(5)] for y in range(10)]
	new_embs = torch.tensor(new_embs, dtype=torch.float)
	sd['embeddings.weight'] = new_embs
	model.state_dict = sd
	print(model.state_dict)
	'''

	split_0 = False

	tag_embs_fixed = [[1 for x in range(5)] for y in range(5)]
	tag_embs_fixed = torch.tensor(tag_embs_fixed, dtype=torch.float)
	sd = model.state_dict()
	sd['tag_embs.weight'] = tag_embs_fixed
	model.state_dict = sd

	if split_0:
		tag_ids = torch.tensor(list(range(5)), dtype=torch.long)
		tag_ids.unsqueeze(0)
		tag_ids = tag_ids.reshape(-1, 1)
		ques_ids = torch.tensor(list(range(5, 20)), dtype=torch.long)
		ques_ids.unsqueeze(0)
		ques_ids = ques_ids.reshape(-1, 1)
	else:
		item_ids = torch.tensor(list(range(20)), dtype=torch.long)
		item_ids.unsqueeze(0)
		item_ids = item_ids.reshape(-1, 1)

	#print(tag_ids)
	#print(ques_ids)
	#y_hat = model(tag_ids, ques_ids)

	y = torch.tensor([[0,0,0,0,1]]*10 + [[1,0,0,0,0]]*10, dtype=torch.float)
	

	optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-8)
	criterion = nn.BCELoss()

	for i in range(5):
		print('epoch:', i)
		model.train()
		model.zero_grad()
		if split_0:
			outputs = model(tag_ids, ques_ids)
		else:
			outputs = model(item_ids)
		#print(outputs)
		#print(y)
		loss = criterion(outputs, y)
		print('loss:', loss)
		loss.backward()
		optimizer.step()
		sd = model.state_dict
		print(sd)
		print(model.parameters)


	





























