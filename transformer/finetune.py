"""
Helper functions for finetuning and existning transformer model
"""

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
