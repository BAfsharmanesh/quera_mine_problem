import random

import numpy as np
import torch
from torch.utils import data
from torchvision.transforms.functional import convert_image_dtype
import torchvision.transforms as transforms


class Dataset_from_memory(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, all_img_tensor, targets_dict_list=None, transform=None, imgBoxTransform=None, train=True):
        super(Dataset_from_memory, self).__init__()
        "Initialization"
        self.targets_dict_list = targets_dict_list
        self.all_img_tensor = all_img_tensor
        self.transform = transform
        self.imgBoxTransform = imgBoxTransform
        self.train = train

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.all_img_tensor)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Load data
        X = self.all_img_tensor[index]
        X = convert_image_dtype(X, dtype=torch.float32)

        if self.train == True:
          if self.transform is not None:
            X = self.transform(X)    

          y = self.targets_dict_list[index]     # (input) spatial images

          if self.imgBoxTransform is not None:
            X, y = self.imgBoxTransform(X, y)

          return X, y          
        
        if self.train == False:
          if self.transform is not None:
            X_r = self.transform(X.clone())      

          return X,X_r, index


def valid_train_split(sample_len, valid_percent=15):
  indx_list = [i for i in range(sample_len)]
  valid_indx = random.choices(indx_list, k=round(valid_percent/100*len(indx_list)))
  train_indx = [i for i in indx_list if i not in valid_indx]
  print(f'valid size: {len(valid_indx)}, train size: {len(train_indx)}')
  return train_indx, valid_indx


class RandomHorizontalFlip(object):

    """Randomly horizontally flips the Image with the probability *p*
    Parameters
    ----------
    p: float
        The probability with which the image is flipped
    Returns
    -------
    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
    """

    def __init__(self, p=0.5):
        self.p = p

    def apply_transform_on_sample(self, img, bboxes):
            img_width = torch.from_numpy(np.array(img.shape[-1]))
            box_width = torch.sub(bboxes[:,2], bboxes[:,0])
            if random.random() < self.p:
                # img = torch.from_numpy(np.array(img)[:, ::-1, :])
                img = img.flip(2)
                bboxes[:, [0, 2]] = img_width - bboxes[:, [0, 2]]
                bboxes[:, 0] -= box_width
                bboxes[:, 2] += box_width
            return img, bboxes

    def __call__(self, img, bboxes):
            img, bboxes['boxes'] = self.apply_transform_on_sample(img, bboxes['boxes']) 
            return img, bboxes
    

def img_transform():
    transforms_list = []
    transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                            )
    transforms_list.append(transforms.Grayscale(num_output_channels=1))
    return transforms.Compose(transforms_list)