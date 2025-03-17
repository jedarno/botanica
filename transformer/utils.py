"""
Useful functions for training transformers
"""
import timm

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



