"""
Helper functions for finetuning and existning transformer model
"""
import numpy as np
import time
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision

from timm.loss import LabelSmoothingCrossEntropy
from torch.optim import lr_scheduler
from tqdm import tqdm

def load_ViT_tf_model(path):
  """
  Function to load in a seved set of ViT weights to be funetuned.

  parameters
  ----------

  path : string 
    file path of a .pt file containing a set of weights for a ViT transformer model

  returns
  -------
  torch.nn.module : model
    PyTorch ViT model with weighrs loaded form the passed .pt file

  list : tune_params
    A list of parameter dictionaries that can be given as a paramter to a torch.Optimiser to finetune 
  """
  model = torch.load(path)

  for param in model.parameters():
    param.requires_grad = False

  for param in model.encoder.layers.encoder_layer_11.parameters():
    param.requires_grad = True

  for param in model.heads.head.parameters():
    param.requires_grad = True

  tune_params = [{'params' : model.encoder.layers.encoder_layer_11.parameters()}, {'params':  model.heads.head.parameters()}]

  return (model, tune_params)

def tune_model(model, param_dict, hyperparams, epochs, scheduler_step_size, scheduler_gamma):
  """
  Function to finetune selected model paramters

  parameters
  ----------
  model : torch.nn.module 
    The model to be finetunes

  param_dict : list 
    List of dictionaries of selected parameters from model sections

  hyperparams : np.ndarray 
    Model hyperparameters for learning

  epochs : int
    Number of epochs over the training data to finetune the model parameters
  
  scheduler_step_size : int
    Step size for multiplicative learning rate scheduler

  scheduler_gamma : int 
    Scheduler gamma value to multiply with the learning rate each scheduler step. 

  returns
  -------
  torch.nn.module
    finetuned model 
  """

  lr=hyperparams[0]
  beta_1 = hyperparams[1]
  decay = hyperparams[2]
  criterion = LabelSmoothingCrossEntropy()
  criterion = criterion.to(device)
  optimizer = optim.AdamW(param_dict, lr=lr, betas = (beta_1, 0.999), weight_decay=decay)
  exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheculer_step_siez, gamma=scheduler_gamma)

  model = get_train_model_val(model, criterion, optimizer, exp_lr_scheduler, num_epochs=epochs)

  return model


