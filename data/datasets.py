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

    negative_class = anchor_class+1 % 2
    negative_index = self.class_indx[negative_class][np.random.randint(0, self.class_indx[negative_class].shape[0]-1)]
    negative_path = self.samples[negative_index][0]
    negative_image = self.loader(negative_path)

    if self.transform is not None:
      positive_image = self.transform(positive_image)
      negative_image = self.transform(negative_image)

    #Match the TripletMarginLoss format (a,p,n)
    return anchor_image, positive_image, negative_image


