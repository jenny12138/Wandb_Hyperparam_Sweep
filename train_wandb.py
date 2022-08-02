import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import time, os, glob
from tqdm import tqdm
from dataloader import GoDataset
from model import RNN
import wandb
import pickle

def reset_wandb_env():
	"""
	Resets the wandb environment such that parallel seeds can be run, taken from 
	https://github.com/wandb/examples/tree/d82e9f047a4c2abcf4234f90c513cac83d1d94ca/examples/wandb-sweeps/sweeps-cross-validation
	"""
	exclude = {
		"WANDB_PROJECT",
		"WANDB_ENTITY",
		"WANDB_API_KEY",
	}
	for k, v in os.environ.items():
		if k.startswith("WANDB_") and k not in exclude:
			del os.environ[k]

def get_loss(y, y_hat, loss_mask):
	"""
	Calculates MSE loss. It calculates the MSE of y and y_hat at EVERY time step, then sums it up
	for the final loss.

	y_hat should have the form (sequence length/tdim=100, batch_size=100 by default, output dimensionality = 33)
	y should have the form (sequence length/tdim=100, batch_size=100 by default, output dimensionality = 33)
	loss_mask should have the form (sequence length/tdim=100, batch_size=100 by default, output dimensionality=33)
	"""
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	mse = torch.mean(torch.square((y - y_hat)) * loss_mask).to(device)

	return mse

def load_data_model(task='go', mode='BPTT', hidden_size=20, train_portion=0.8,batch_size=100,non_linearity='tanh',noise='sparse',device='cpu', tdim=100):
	"""
	Generates training dataloader, testing dataloader, and the RNN model with the appropriate learning rule
	"""
	assert mode in ['BPTT','RTRL','eprop'], "Not implemented for mode {}".format(mode)
	if (task=='go'):
		dataset = GoDataset(data_folder="/home/mila/j/jiayue.zheng/Projects/BP2T2/Go/scratch_data") 
	else:
		print("Unrecognized task")
	train_size = int(train_portion*len(dataset))
	test_size = len(dataset) - train_size
	train_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_size,test_size])
	train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
	test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
	if mode=='BPTT':
		model = RNN(hidden_size=hidden_size,non_linearity=non_linearity,device=device, tdim=tdim).to(device)

	return train_dataloader, test_dataloader, model

def BPTT_train(task='go', mode='BPTT',hidden_size=20,n_epochs=10,batch_size=20,non_linearity='tanh',lr=1e-3,momentum=0.0,gc=100, save_results=False, tdim=100):
	"""
	Trains an RNN using Back Propagation Through Time
	"""
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	train_dataloader, test_dataloader, model = load_data_model(task=task, mode=mode, hidden_size=hidden_size,
		batch_size=batch_size,non_linearity=non_linearity,device=device, tdim=tdim)
	#loss_func = torch.nn.MSELoss()
	optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum)
	tr_loss_arr = []
	te_loss_arr = []
	trivial_loss_arr = []

	model.eval()
	with torch.no_grad():
		tot_loss = 0
		for X,y,loss_mask in test_dataloader:
			X = X.to(device) # (batch_size, tdim, input_dimensionality)
			y = y.to(device) # (batch_size, tdim, output_dimensionality)
			loss_mask = loss_mask.to(device) # (batch_size, tdim, output_dimensionality)
			X = torch.transpose(X,0,1) # (tdim, batch_size, input_dimensionality)
			y = torch.transpose(y,0,1) # (tdim, batch_size, output_dimensionality)
			loss_mask = torch.transpose(loss_mask,0,1) # (tdim, batch_size, output_dimensionality)
			out = model(X)
			loss = get_loss(y, out, loss_mask)
			tot_loss += loss.item()
	trivial_loss_arr.append(tot_loss/len(test_dataloader))
	trivial_loss_arr = np.array(trivial_loss_arr)

	for epoch in tqdm(range(n_epochs)):
		model.train() # turn on training mode
		tot_loss = 0
		for X,y,loss_mask in train_dataloader:
			X = X.to(device) # (batch_size, tdim, input_dimensionality)
			y = y.to(device) # (batch_size, tdim, output_dimensionality)
			loss_mask = loss_mask.to(device) # (batch_size, tdim, output_dimensionality)
			X = torch.transpose(X,0,1) # (tdim, batch_size, input_dimensionality)
			y = torch.transpose(y,0,1) # (tdim, batch_size, output_dimensionality)
			loss_mask = torch.transpose(loss_mask,0,1) # (tdim, batch_size, output_dimensionality)
			optimizer.zero_grad()
			out = model(X)
			loss = get_loss(y, out, loss_mask)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(),gc)
			optimizer.step()
			tot_loss += loss.item()

		tr_loss = tot_loss/len(train_dataloader)
		tr_loss_arr.append(tr_loss)

		model.eval() # turn on validation/testing mode (no backward call, since we're testing)
		with torch.no_grad():
			tot_loss = 0
			for X,y,loss_mask in test_dataloader:
				X = X.to(device) # (batch_size, tdim, input_dimensionality)
				y = y.to(device) # (batch_size, tdim, output_dimensionality)
				loss_mask = loss_mask.to(device) # (batch_size, tdim, output_dimensionality)
				X = torch.transpose(X,0,1) # (tdim, batch_size, input_dimensionality)
				y = torch.transpose(y,0,1) # (tdim, batch_size, output_dimensionality)
				loss_mask = torch.transpose(loss_mask,0,1) # (tdim, batch_size, output_dimensionality)
				out = model(X)
				loss = get_loss(y, out, loss_mask)
				tot_loss += loss.item()

			# Get a few results printed out to see whether learning occured, at the very end
			"""
			if (epoch == n_epochs-1):
				for X,y,loss_mask in test_dataloader:
					X = X.to(device) # (batch_size, tdim, input_dimensionality)
					y = y.to(device) # (batch_size, tdim, output_dimensionality)
					loss_mask = loss_mask.to(device) # (batch_size, tdim, output_dimensionality)
					X = torch.transpose(X,0,1) # (tdim, batch_size, input_dimensionality)
					y = torch.transpose(y,0,1) # (tdim, batch_size, output_dimensionality)
					loss_mask = torch.transpose(loss_mask,0,1) # (tdim, batch_size, output_dimensionality)
					#print("X is: ", X[:,[0,1],:], "\n") # print out 2 examples of inputs
					out = model(X[:,[0,1],:])
					#print("output is: ", out, "\n") # print out 2 examples of the corresponding output
					#print("y is: ", y[:,[0,1],:], "\n") # print out 2 examples of y
					out = model(X)
					check_dict = {"X": X, "out": out, "y":y}
					pickle_name = "/home/mila/j/jiayue.zheng/Projects/BP2T2/Go/scratch_data/check_results_{}_{}.pickle".format(task, mode)
					pickle.dump(check_dict,open(pickle_name,"wb"))
					break
			"""
			
		te_loss = tot_loss/len(test_dataloader)
		te_loss_arr.append(te_loss)

		if save_results == False:
			run.log({"train_loss": tr_loss, "test_loss":te_loss})

	return trivial_loss_arr.mean(), tr_loss_arr, te_loss_arr

if __name__=='__main__':
	# Take inputs
	parser = argparse.ArgumentParser(description="Testing BPTT on the Go task")
	parser.add_argument("--task",type=str,default='go',help="Cognitive task being peformed,e.g. Go")
	parser.add_argument("--hidden_dim",type=int,default=20,help="Hidden dimension size")
	parser.add_argument("--non_linearity",type=str,default='relu',help="RNN non-linearity")
	parser.add_argument("--mode",type=str,default='BPTT',help="Temporal credit assignment algorithm, e.g. BPTT")
	parser.add_argument("--noise",type=str,default='sparse',help="Type of noise to be added to backward weights")
	parser.add_argument("--epochs",type=int,default=500,help="Number of training epochs")
	parser.add_argument("--lr",type=float,default=1e-3,help="Learning rate value")
	parser.add_argument("--momentum",type=float,default=0.0,help="Momentum value")
	parser.add_argument("--gc",type=int,default=100,help="Gradient clipping value")
	parser.add_argument("--fold",type=int,default=5,help="Which fold to run")
	parser.add_argument("--sweep_id",type=str,default='unknown',help="ID of the sweep agent") 
	parser.add_argument("--sweep_run_name",type=str,default='unknown',help="Name of the sweep")
	parser.add_argument("--save_results",default=False,action='store_true',help="Add to save plots and results in file, else hyperparameter search with Wandb is run")
	ns, unknown_args = parser.parse_known_args()
	if len(unknown_args)==0:
		args = parser.parse_args()
		argsdict = args.__dict__
	else:
		import sys, yaml
		assert sys.argv.index('--script-config')>=0, "Need to either pass the script config file as --script-config <filename>"
		config_fname = sys.argv[sys.argv.index('--script-config')+1]
		argsdict = yaml.load(open(config_fname,'r'),Loader=yaml.FullLoader)

	# Loading arguments into variabls
	print(argsdict)
	tdim = 100 # 20ms * 100 = 2000ms experiment
	dt = 20
	task = argsdict["task"]
	hidden_dim = argsdict["hidden_dim"]
	hidden_dim_text = "_hiddenDim{}".format(hidden_dim) if hidden_dim!=20 else ""
	non_linearity = argsdict["non_linearity"]
	mode = argsdict["mode"]
	noise = argsdict["noise"]
	n_epochs = argsdict["epochs"]
	lr = argsdict["lr"]
	momentum = argsdict["momentum"]
	gc = argsdict["gc"]
	save_results = argsdict["save_results"]
	fold = argsdict["fold"]
	sweep_id = argsdict["sweep_id"] 
	sweep_run_name = argsdict["sweep_run_name"] 

	# Make preparations, depending on which mode of save_folder is ran
	save_folder = 'Results_{}_{}{}_gc{}'.format(non_linearity,tdim,hidden_dim_text,gc)
	if save_results:
		if not os.path.exists(save_folder):
			os.makedirs(save_folder)
	else:
		reset_wandb_env()
		run_name = "{}-{}".format(sweep_run_name, fold)
		run = wandb.init(
			group=sweep_id,
			job_type=sweep_run_name,
			name=run_name,
			config=argsdict
		)
		run = wandb.init(settings=wandb.Settings(start_method="fork")) # Set starting method to avoid well-documented error: https://docs.wandb.ai/guides/track/launch#init-start-error
	
	tr_loss_arr_folds = []
	te_loss_arr_folds = []
	trivial_loss_folds = []

	np.random.seed(fold) 
	torch.manual_seed(fold)

	if mode=='BPTT':
		trivial_loss, BPTT_tr_loss_arr, BPTT_te_loss_arr = BPTT_train(task=task, mode='BPTT',hidden_size=hidden_dim,n_epochs=n_epochs,
			batch_size=100, non_linearity=non_linearity, lr=lr, momentum=momentum, gc=gc,save_results=save_results, tdim=tdim)
		trivial_loss_folds.append(trivial_loss)
		tr_loss_arr_folds.append(BPTT_tr_loss_arr)
		te_loss_arr_folds.append(BPTT_te_loss_arr)

	# Only save results if the best hyperparameters have already been determined. Otherwise, record the loss but not the hyperparameters.
	if save_results:
		np.savez(os.path.join(save_folder,"avg_{}{}_performance_{}_{}{}.npz".format(mode,noise if 'BP2T2' in mode else "",tdim,non_linearity,hidden_dim_text)),
			trivial_loss=np.array(trivial_loss_folds), tr_loss_arr=np.array(tr_loss_arr_folds), te_loss_arr=np.array(te_loss_arr_folds))

		plt.figure()
		if len(trivial_loss_folds)==1:
			plt.plot(np.linspace(1,len(tr_loss_arr_folds[0]),len(tr_loss_arr_folds[0])),tr_loss_arr_folds[0],'b-',label='{} {} train'.format(mode,non_linearity))
			plt.plot(np.linspace(1,len(te_loss_arr_folds[0]),len(te_loss_arr_folds[0])),te_loss_arr_folds[0],'b--',label='{} {} test'.format(mode,non_linearity))
		else:
			tr_loss_arr_folds = np.array(tr_loss_arr_folds)
			te_loss_arr_folds = np.array(te_loss_arr_folds)
			plt.errorbar(x=np.linspace(1,len(tr_loss_arr_folds[0]),len(tr_loss_arr_folds[0])),y=np.nanmean(tr_loss_arr_folds,0),yerr=np.std(tr_loss_arr_folds,0),
				color='b',alpha=0.4,ls='-',label='{} {} train'.format(mode,non_linearity))
			plt.errorbar(x=np.linspace(1,len(te_loss_arr_folds[0]),len(te_loss_arr_folds[0])),y=np.nanmean(te_loss_arr_folds,0),yerr=np.std(te_loss_arr_folds,0),
				color='r',alpha=0.4,ls='--',label='{} {} test'.format(mode,non_linearity))
			plt.plot(np.linspace(1,len(tr_loss_arr_folds[0]),len(tr_loss_arr_folds[0])),np.nanmean(tr_loss_arr_folds,0),color='b',ls='-')
			plt.plot(np.linspace(1,len(te_loss_arr_folds[0]),len(te_loss_arr_folds[0])),np.nanmean(te_loss_arr_folds,0),color='r',ls='--')

		plt.legend()
		plt.ylim([0,0.25])
		plt.savefig(os.path.join(save_folder,"avg_plot_{}{}_{}_{}{}.png".format(mode,noise if 'BP2T2' in mode else "",tdim,non_linearity,hidden_dim_text)))

	else:
		if len(trivial_loss_folds)==1:
			res_report = np.nansum(te_loss_arr_folds[0])+trivial_loss_folds[0]
		else:
			te_loss_arr_folds = np.array(te_loss_arr_folds)
			trivial_loss_folds = np.array(trivial_loss_folds)
			mean_te_loss_arr_folds = np.nanmean(te_loss_arr_folds,0)
			res_report = np.nansum(mean_te_loss_arr_folds)+np.nanmean(trivial_loss_folds)