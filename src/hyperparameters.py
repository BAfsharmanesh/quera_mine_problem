from dataset import Dataset_from_memory, RandomHorizontalFlip, img_transform
from train import EarlyStopper, train_one_epoch, valid_one_epoch
from model import model_select
from utils import collate_fn

from tqdm.auto import tqdm
import torch
from torch.utils import data
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, faster_rcnn
from torchvision.models import ResNet50_Weights

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from scipy.stats import uniform, randint


def objective_hyperopt(inputData, lr=0.05, momentum=0.9, weight_decay=0.0005, batch_size=8, patience=3, num_classes=2):
  
  targets_dict_list, all_img_tensor, train_indx, valid_indx = inputData
  
  conf_params = {'drop_last':True, 'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True, 'collate_fn':collate_fn}
  train_set =   Dataset_from_memory([targets_dict_list[i] for i in train_indx], all_img_tensor[train_indx,:,:,:], 
                                    transform=img_transform(),
                                    imgBoxTransform=RandomHorizontalFlip(0.5),
                                    train=True)
  train_loader = data.DataLoader(train_set, **conf_params)

  valid_set =   Dataset_from_memory([targets_dict_list[i] for i in valid_indx], all_img_tensor[valid_indx,:,:,:], 
                                    transform=img_transform(),
                                    imgBoxTransform=None,
                                    train=True)
  valid_loader = data.DataLoader(valid_set, **conf_params)

  num_epochs = 40

  model, optimizer, device = model_select(lr, momentum, weight_decay, num_classes) # lr_scheduler

  trainingEpoch_loss = []
  validationEpoch_loss = []
  early_stopper = EarlyStopper(patience=patience, min_delta=0.01)

  for epoch in tqdm(range(num_epochs)):
    try:
      train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, log_interval=10, verbose=True)
      valid_loss = valid_one_epoch(model, valid_loader, device, epoch, verbose=True)
      # lr_scheduler.step(valid_loss)
      trainingEpoch_loss.append(train_loss)
      validationEpoch_loss.append(valid_loss)
      if early_stopper.early_stop(valid_loss):             
          return min(validationEpoch_loss)
          break
    except:
      print('error')
      return 1
  return min(validationEpoch_loss)



def hyperopt_fmin(inputData):

    def objective(space):
        print(space)
        loss = objective_hyperopt(inputData,
                        lr=space['lr'], 
                        momentum=space['momentum'], 
                        weight_decay=space['weight_decay'], 
                        batch_size=8,
                        patience=3,
                        num_classes=2,
                        )
        print ("SCORE:", loss)
        return {'loss': loss, 'status': STATUS_OK }

    trials = Trials()
    space={'lr' : hp.uniform('lr', 0,0.01),
            'momentum' : hp.uniform('momentum', 0.85,0.99),
            'weight_decay':hp.uniform('weight_decay', 0,0.01),
        }
    best_hyperparams = fmin(fn = objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = 50,
                            trials = trials)
    return best_hyperparams, trials