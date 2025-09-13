"""
Useful functions for training transformers
"""
import copy
import time
import timm
import torch
import torch.optim as optim
import torch.nn as nn

from timm.loss import LabelSmoothingCrossEntropy
from torchvision import datasets, models
from torchvision import transforms as T

def get_transforms(*, data):

  if data == 'train':
    return T.Compose([
      T.RandomHorizontalFlip(),
      T.RandomVerticalFlip(),
      T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.25),
      T.Resize(256),
      T.CenterCrop(224),
      T.ToTensor(),
      T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD), # imagenet means
      T.RandomErasing(p=0.1, value='random')
    ])

  elif data == 'valid':
    return T.Compose([ # We dont need augmentation for test transforms
      T.Resize(256),
      T.CenterCrop(224),
      T.ToTensor(),
      T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD), # imagenet means
    ])

def get_train_model(model, criterion, optimizer, scheduler, trainloader, trainset, valloader, valset, device, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = trainloader
                size = len(trainset)
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = valloader
                size = len(valset)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / size
            epoch_acc = running_corrects / size

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def train_model_loss(model, criterion, optimizer, scheduler, trainloader, trainset, valloader, valset, device, num_epochs=1):
    """

    Args:
      model: Architecture and its pretrained weights
      criterion: The loss function
      optimizer: Optimisation algorithm
      scheduler: Learning rate Scheduler
      num_epochs: How many epochs model is trained for

    Returns: The validation loss of the model.

    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = trainloader
                size = len(trainset)
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = valloader
                size = len(valset)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / size
            epoch_acc = running_corrects / size

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return epoch_loss

def train_model_wrapper_vit_b(params, trainloader, trainset, valloader, valset, device, num_epochs):
  """
  params is list of parameters to optimise
  params[0] = Learning rate of optimiser
  params[1] = gamma of schedular
  """

  classes = trainset.classes
  model = models.vit_b_16(weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)

  for param in model.parameters():
    param.requires_grad = False

  n_inputs = model.heads.head.in_features
  model.heads.head = nn.Sequential(
    nn.Linear(n_inputs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(classes))
  )
  model = model.to(device)

  criterion = LabelSmoothingCrossEntropy()
  criterion = criterion.to(device)
  optimizer = optim.AdamW(model.heads.head.parameters(), lr=params[0], betas = (params[1], 0.999), weight_decay=params[2])
  exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

  model = get_train_model(model, criterion, optimizer, exp_lr_scheduler, trainloader, trainset, valloader, valset, device, num_epochs=num_epochs)

  return model

def train_model_wrapper_regnety16gf(params, trainloader, trainset, valloader, valset, device, num_epochs, classes=None):
  """
  params is list of parameters to optimise
  params[0] = Learning rate of optimiser
  params[1] = gamma of schedular
  """
  
  if classes == None:
    classes = trainset.classes

  model = models.regnet_y_16gf(weights = models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_LINEAR_V1).to(device)

  for param in model.parameters():
    param.requires_grad = False

  n_inputs = model.fc.in_features
  model.fc = nn.Sequential(
    nn.Linear(n_inputs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(classes))
   )
  for param in model.trunk_output.block3.parameters():
    param.requires_grad = True

  for param in model.trunk_output.block4.parameters():
    param.requires_grad = True

  tune_params = [{'params' : model.trunk_output.block3.parameters()}, {'params' : model.trunk_output.block4.parameters()}, {'params' : model.fc.parameters()}]

  model = model.to(device)

  criterion = LabelSmoothingCrossEntropy()
  criterion = criterion.to(device)
  optimizer = optim.AdamW(tune_params, lr=params[0], betas = (params[1], 0.999), weight_decay=params[2])
  exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
  loss = train_model_loss(model, criterion, optimizer, exp_lr_scheduler, trainloader, trainset, valloader, valset, device, num_epochs=num_epochs)

  return loss

def train_model_wrapper_vit_l(params, trainloader, trainset, valloader, valset, device, num_epochs, classes=None):
  """
  params is list of parameters to optimise
  params[0] = Learning rate of optimiser
  params[1] = gamma of schedular
  """
  if classes == None:
    classes = trainset.classes

  model = models.vit_l_16(weights = models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)

  for param in model.parameters():
    param.requires_grad = False

  n_inputs = model.heads.head.in_features
  model.heads.head = nn.Sequential(
    nn.Linear(n_inputs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(classes))
  )

  for param in model.encoder.layers.encoder_layer_23.parameters():
    param.requires_grad = True

  for param in model.encoder.ln.parameters():
    param.requires_grad = True

  for param in model.heads.head.parameters():
    param.requires_grad = True

  tune_params = [{'params' : model.encoder.layers.encoder_layer_23.parameters()}, {'params': model.encoder.ln.parameters()}, {'params':  model.heads.head.parameters()}]
  model = model.to(device)

  criterion = LabelSmoothingCrossEntropy()
  criterion = criterion.to(device)
  optimizer = optim.AdamW(tune_params, lr=params[0], betas = (params[1], 0.999), weight_decay=params[2])
  exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
  loss = train_model_loss(model, criterion, optimizer, exp_lr_scheduler, trainloader, trainset, valloader, valset, device, num_epochs=num_epochs)

  return loss

def train_model_wrapper_swin_b(params, trainloader, trainset, valloader, valset, device, num_epochs):
  """
  params is list of parameters to optimise
  params[0] = Learning rate of optimiser
  params[1] = gamma of schedular
  """
  
  classes = trainset.classes
  model = models.swin_b(weights = models.Swin_B_Weights.IMAGENET1K_V1)

  for param in model.parameters():
    param.requires_grad = False

  n_inputs = model.head.in_features
  model.head = nn.Sequential(
    nn.Linear(n_inputs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(classes))
   )
  model = model.to(device)

  criterion = LabelSmoothingCrossEntropy()
  criterion = criterion.to(device)
  optimizer = optim.AdamW(model.head.parameters(), lr=params[0], betas = (params[1], 0.999), weight_decay=params[2])
  exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

  model = get_train_model(model, criterion, optimizer, exp_lr_scheduler, trainloader, trainset, valloader, valset, device, num_epochs=num_epochs)


  return model

def triplet_train_model(model, criterion, optimizer, scheduler, trainloader, trainset, valloader, valset, device, num_epochs=1, n_shot=3):
    """

    Args:
      model: Architecture and its pretrained weights
      criterion: The loss function
      optimizer: Optimisation algorithm
      scheduler: Learning rate Scheduler
      num_epochs: How many epochs model is trained for

    Returns: The trained model
    """

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
      print(f'Epoch {epoch}/{num_epochs - 1}')
      print('-' * 10)

      #Perform train and val stages for each epoch
      for phase in ['train', 'val']:
        if phase == 'train':
          model.train()  # Set model to training mode
          dataloader = trainloader
          size = len(trainset)
        else:
          model.eval()   # Set model to evaluate mode
          dataloader = valloader
          size = len(valset)

        running_loss = 0.0

        #iterating over batch
        for anchor, pos, neg, _  in dataloader:
          anchor = anchor.to(device)
          pos = pos.to(device)
          neg = neg.to(device)

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward
          anchor_embedding, positive_embedding, negative_embedding = model(anchor, pos, neg)
          # track history if only in train
          with torch.set_grad_enabled(phase == 'train'):
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)

            #backwards + optimise only if training
            if phase == 'train':
              loss.backward()
              optimizer.step()

          #statistics
          running_loss += loss.item() * anchor.size(0)

        if phase == 'train':
          scheduler.step()

        epoch_loss = running_loss / size

        print(f'{phase} Loss: {epoch_loss:.4f}')

        # deep copy the model
        if phase == 'val' and epoch_loss < best_loss:
          best_loss = epoch_loss
          best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
