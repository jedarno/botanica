import numpy as np
import os
import os.path
from pathlib import Path
from typing import Any, Callable, cast, Optional, Union

from PIL import Image
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class BinarySiameseImageFolder(DatasetFolder):

  def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        ):

        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
        self.get_class_index()
    
  def get_class_index(self):
    np_targets = np.array(self.targets)
    self.class_indx = {}

    for i in range(0,2):
      self.class_indx[i] = np.where(np_targets == i)[0]


  def __getitem__(self, index):
    """
    Return a tuple containing an anchor image and positoive and negative examples (anchor, pos, neg)
    """
    anchor_image, anchor_class = super().__getitem__(index)

    #Positive and negative examples
    positive_index = self.class_indx[anchor_class][np.random.randint(0, self.class_indx[anchor_class].shape[0]-1)]

    while positive_index == index:
      positive_index = self.class_indx[anchor_class][np.random.randint(0, self.class_indx[anchor_class].shape[0]-1)]

    positive_path = self.samples[positive_index][0]
    positive_image = self.loader(positive_path)

    if anchor_class == 0:
      negative_class = 1
    elif anchor_class == 1:
      negative_class = 0

    negative_index = self.class_indx[negative_class][np.random.randint(0, self.class_indx[negative_class].shape[0]-1)]
    negative_path = self.samples[negative_index][0]
    negative_image = self.loader(negative_path)

    if self.transform is not None:
      positive_image = self.transform(positive_image)
      negative_image = self.transform(negative_image)

    #Match the TripletMarginLoss format (a,p,n)
    return anchor_image, positive_image, negative_image

  def get_support_set(self, n_shot)
    """
    Return image plus a support set for both classes
    """

    #support sets
    #For first class
    class1_support_set = []
    class2_support_set = []
    for n in range(self.n_shot):
      image_indx1 = self.class_indx[0][np.random.randint(0, self.class_indx[0].shape[0]-1)] 

      while image_indx1 == index:
        image_indx1 = self.class_indx[0][np.random.randint(0, self.class_indx[0].shape[0]-1)]  

      image_indx2 = self.class_indx[1][np.random.randint(0, self.class_indx[1].shape[0]-1)] 

      image_path1 = self.samples[image_indx1][0]
      image_path2 = self.samples[image_indx2][0]
      image1 = self.loader(image_path1)
      iamge2 = self.loader(image_path2)

      if self.transform is not None:
        image1 = self.transform(image1)
        image2 = self.transform(iamge2)

      class1_support_set.append(image1)
      class2_support_set.append(image2)

    return class1_support_set, class2_support_set

class BinarySupportSet(DatasetFolder):

  def __init__(
        self,
        root: Union[str, Path],
        n_shot = 3,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        ):

        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
        self.n_shot = n_shot
        self.get_class_index()
    
  def get_class_index(self):
    np_targets = np.array(self.targets)
    self.class_indx = {}

    for i in range(0,2):
      self.class_indx[i] = np.where(np_targets == i)[0]

  def __getitem__(self, index):
    """
    Return image plus a support set for both classes
    """

    #support sets
    #For first class
    class1_support_set = []
    class2_support_set = []
    for n in range(self.n_shot):
      image_indx1 = self.class_indx[0][np.random.randint(0, self.class_indx[0].shape[0]-1)] 

      while image_indx1 == index:
        image_indx1 = self.class_indx[0][np.random.randint(0, self.class_indx[0].shape[0]-1)]  

      image_indx2 = self.class_indx[1][np.random.randint(0, self.class_indx[1].shape[0]-1)] 

      image_path1 = self.samples[image_indx1][0]
      image_path2 = self.samples[image_indx2][0]
      image1 = self.loader(image_path1)
      iamge2 = self.loader(image_path2)

      if self.transform is not None:
        image1 = self.transform(image1)
        image2 = self.transform(iamge2)

      class1_support_set.append(image1)
      class2_support_set.append(image2)

    return class1_support_set, class2_support_set

      
