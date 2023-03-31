import os
import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import convert_image_dtype


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(12, 9))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def boxs_labels_fun(img_int, img_nm, labels_df):
  selec_labels = labels_df[labels_df.image_name == img_nm]
  imgf_h, imgf_w = img_int.shape[-2:]
  img_h, img_w = selec_labels.loc[:,['image_height', 'image_width']].values[0]

  rw = imgf_w/img_w
  rh = imgf_h/img_h

  img_LBS = selec_labels.loc[:,['xmin',	'ymin',	'width',	'height']].values
  img_LBS[:,2] = img_LBS[:,0] + img_LBS[:,2]
  img_LBS[:,3] = img_LBS[:,1] + img_LBS[:,3]

  img_LBS[:,0] = img_LBS[:,0]*rw
  img_LBS[:,2] = img_LBS[:,2]*rw

  img_LBS[:,1] = img_LBS[:,1]*rh
  img_LBS[:,3] = img_LBS[:,3]*rh

  label = [0 if lab =='wood' else 1 for lab in selec_labels.label_name.values]
  return torch.FloatTensor(img_LBS), torch.LongTensor(label), rw, rh


def show_bounding_boxe(data_path, img_nm, labels_df, from_ram=False): 
  if from_ram:
    img_int = data_path
  else:
    img_int = read_image(data_path + img_nm)

  boxes, colors, rw, rh = boxs_labels_fun(img_int, img_nm, labels_df)
  colors = ['blue' if i ==0 else 'yellow' for i in colors]
  result = draw_bounding_boxes(img_int, boxes, colors=colors, width=int(20*rw))
  show(result)


def img_tensor_load(data_path, df, transform=False):
  img_list = []
  for img_fl in tqdm( df.File ):     
    # Imgs
    image = read_image(data_path+img_fl)
    if transform:
      image = transform(image)
    img_list.append(image)
  return torch.stack( img_list )


def load_weights_fun(path, model, optimizer):
    loaded_chck = torch.load(os.path.join(path), map_location=torch.device('cpu'))
    model.load_state_dict(loaded_chck['R_CNN'])
    optimizer.load_state_dict(loaded_chck['optimizer'])
    print('epoch:', loaded_chck['epoch'],'valid_loss:',loaded_chck['valid_loss'],'train_loss:',loaded_chck['train_loss']) 
    print('CRNN model reloaded!')


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def imgs_with_boxes(X, y, score_threshold=0.8, pred=True):
  if pred:
    imgs_with_boxes = [
                            draw_bounding_boxes(img_int, boxes=output['boxes'][output['scores'] > score_threshold], width=2, colors='yellow')
                            for img_int, output in zip([convert_image_dtype(x.to('cpu'), dtype=torch.uint8) for x in X], y)
                            ]
  else:
    imgs_with_boxes = [
                            draw_bounding_boxes(img_int, boxes=output['boxes'], width=2, colors=['yellow' if i==1 else 'blue' for i in output['labels'] ])
                            for img_int, output in zip([convert_image_dtype(x.to('cpu'), dtype=torch.uint8) for x in X], y)
                        ]
  return  imgs_with_boxes


def collate_fn(batch):
    return tuple(zip(*batch))