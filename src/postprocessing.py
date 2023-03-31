import itertools

import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils import data
import torchvision

from src.dataset import img_transform
from src.utils import load_weights_fun
from src.model import model_select
from src.dataset import Dataset_from_memory


def validation_fun(models, data_loader, device):
  model_wood, model_rock = models['wood'], models['rock']
  model_wood.eval()
  model_rock.eval()
  model_wood.to(device)
  model_rock.to(device)

  output_wood_list = []
  output_rock_list = []
  indx_list = []
  with torch.no_grad():
    for X_w, X_r, indx in tqdm(data_loader):
        # distribute data to device
        X_w = list(img.to(device) for img in X_w)
        X_r = list(img.to(device) for img in X_r)

        output_wood = model_wood(X_w)
        output_rock = model_rock(X_r)

        output_wood_list.append(output_wood)
        output_rock_list.append(output_rock)
        indx_list.append(indx)

  output_wood_list = list(itertools.chain(*output_wood_list))
  output_rock_list = list(itertools.chain(*output_rock_list))
  indx_list = list(itertools.chain(*indx_list))
  
  return output_wood_list, output_rock_list, indx_list


def th_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]


def drop_duplicate_box(boxes):
  boxes_temp = boxes
  same_box_list = []
  for i, box1 in enumerate(boxes_temp[:-1]):
    for j, box2 in enumerate(boxes_temp[i+1:],i+1):
      c1 = torch.abs(box1[1] - box2[1])<3
      c2 = torch.abs(box1[3] - box2[3])<3
      c3 = box1[0] == box2[0]
      c4 = box1[2] == box2[2]
      if c1 and c2 and c3 and c4:
        same_box_list.append([i,j])
  drop_list = list(set([i[1] for i in same_box_list]))
  final_list = [i for i in range(len(boxes_temp)) if i not in drop_list]
  return boxes_temp[final_list,:]


def combine_2_boxes(boxes):
  box1, box2 = boxes
  new_box = torch.Tensor([min(box1[0],box2[0]), min(box1[1],box2[1]), max(box1[2],box2[2]), max(box1[3],box2[3])])
  return new_box


def combine_boxes_1_iter(boxes, device, box_iou_threshold=0.1):
  iou_bxs = torchvision.ops.box_iou(boxes, boxes)
  n_interact_bx = torch.triu(torch.logical_and(iou_bxs >  box_iou_threshold, iou_bxs < 1), diagonal=1).nonzero()
  indx_bxs_wo_interact = th_delete(torch.arange(len(boxes)), n_interact_bx)

  bxs_wo_interact = [x for x in boxes[indx_bxs_wo_interact] ]
  for i in range(len(n_interact_bx)):
    bxs_wo_interact.append( combine_2_boxes(boxes[n_interact_bx][i]) )
  bxs_wo_interact = [x.to(device) for x in bxs_wo_interact]
  return torch.stack(bxs_wo_interact, 0) if len(bxs_wo_interact)!=0 else boxes


def combine_boxes(output_rock_list, device, box_iou_threshold = 0.1):
  
  list_temp = []
  for boxes_i in tqdm(output_rock_list):
    boxes_i = drop_duplicate_box(boxes_i)
    cn=True
    iter_c=0
    bxs_wo_interact = boxes_i
    len_0 = len(bxs_wo_interact)
    while(cn):
      old_len = len(bxs_wo_interact)
      bxs_wo_interact = combine_boxes_1_iter(bxs_wo_interact, device, box_iou_threshold=box_iou_threshold)
      bxs_wo_interact = drop_duplicate_box(bxs_wo_interact)
      if len(bxs_wo_interact) > 0:
        iter_c += 1
        if (len(bxs_wo_interact)>=old_len) and iter_c>len_0:
          cn=False
      else:
        cn=False
    list_temp.append(drop_duplicate_box(bxs_wo_interact))
  return list_temp


def select_rock_fun(rock_box, wood_box, device, sel_st='before'):
  rock_box_temp = rock_box.clone()
  rock_box_temp = rock_box_temp.to(device)
  wood_box = wood_box.to(device)
  rock_box_temp[:,1] = 0.5*(rock_box[:,1] + rock_box[:,3])
  rock_box_temp[:,3] = 0.5*(rock_box[:,1] + rock_box[:,3])

  if sel_st == 'before':
    aaa = torch.where((rock_box_temp[:,[2,3]] < wood_box[2:]).sum(axis=1) == 2, True, False)
    bbb = torch.where(((rock_box_temp[:,[2,3]] > wood_box[[2,1]]) == torch.Tensor([True,  False]).to(device)).sum(axis=1) ==2, True, False)
    return aaa + bbb
  if sel_st == 'after':
    aaa = torch.where((rock_box_temp[:,[2,3]] > wood_box[[2,1]]).sum(axis=1) == 2, True, False)
    bbb = torch.where(((rock_box_temp[:,[2,3]] < wood_box[2:]) == torch.Tensor([True,  False]).to(device)).sum(axis=1) ==2, True, False)
    return aaa + bbb  

def sort_wood(wood_boxes, device):
  wood_boxes = wood_boxes.to(device)
  y_loc_bin = [130, 170, 200, 230, 260, 300]
  wood_sort_list = []
  for i in range(len(y_loc_bin)-1):
    bin_list = []
    for j in wood_boxes:
      if y_loc_bin[i] < j[1] < y_loc_bin[i+1] :
        bin_list.append(j)
    if len(bin_list) != 0:
      wood_sort_list.append(torch.sort(torch.stack(bin_list), 0)[0])
  return torch.cat(wood_sort_list)

def cal_length(boxes):
  l = 0
  for i in boxes:
    l += i[2] - i[0]
  return l/400*110

def calc_runs_length_a_box(rock_boxes, wood_boxes, device):
  run_list = []
  sorted_wood_boxes = sort_wood(wood_boxes, device)
  sel1 = select_rock_fun(rock_boxes, sorted_wood_boxes[0], device, sel_st='before')
  run_list.append( cal_length( rock_boxes[sel1] ) )
  if len(wood_boxes) > 1 :
    for wd_ndx in range(len(wood_boxes)-1):
      sel1 = select_rock_fun(rock_boxes, sorted_wood_boxes[wd_ndx], device, sel_st='after')
      sel2 = select_rock_fun(rock_boxes, sorted_wood_boxes[wd_ndx+1], device, sel_st='after')
      selt = torch.logical_xor(sel1, sel2)
      run_list.append( cal_length( rock_boxes[selt] ) )
  
  sel1 = select_rock_fun(rock_boxes, sorted_wood_boxes[-1], device, sel_st='after')
  run_list.append( cal_length( rock_boxes[sel1] ) )
  return run_list

def calc_runs_length_all_boxes(output_rock_list, output_wood_list, device):
  runs_list_temp = []
  for rock_boxes, wood_boxes in zip(output_rock_list, output_wood_list):
    runs_list_temp.append( calc_runs_length_a_box(rock_boxes, wood_boxes, device) )
  # return run with lenght longer that 10cm
  runs_list = []
  for i in runs_list_temp:
    runs_list.append([j.item() if j >= 10 else 0.0 for j in i])
  return runs_list


def calc_rq(runs_list_dep, runs_list):
  rqs_list = []
  for run_depth, run_L10 in zip(runs_list_dep, runs_list):
    rq_list = []
    for run_L10_k, run_depth_k in zip( run_L10, run_depth ):
      rq_list.append( run_L10_k/run_depth_k*100 )
    
    lenDepth_lenL10 = len(run_depth)-len(run_L10)
    if lenDepth_lenL10 > 0:
      for i in range(lenDepth_lenL10):
        rq_list.append(0)
    rqs_list.append(rq_list)
  return rqs_list


def type_selection(a):
  tp_list = [0, 25, 50, 75, 90, 100, 1000]
  tp_tp = [1, 2, 3, 4, 5, 5]
  for i in range(len(tp_tp)):
    if tp_list[i] <= a < tp_list[i+1]:
      return tp_tp[i]
      break
  if  a > tp_list[-1]:
    print('wrong')


def inference_test_data(all_img_tensor_test, 
                        device,
                        rock_model_path, 
                        wood_model_path,
                        score_threshold=0.8, 
                        box_iou_threshold=0.1):

  conf_params = {'drop_last':False, 'batch_size': 8, 'shuffle': False, 'num_workers': 2, 'pin_memory': True}
  data_set =   Dataset_from_memory(all_img_tensor=all_img_tensor_test, 
                                    transform=img_transform(),
                                    train=False)
  data_loader = data.DataLoader(data_set, **conf_params)


  model_wood, optimizer_wood, device = model_select(lr=0.001, 
                                          momentum=0.99, 
                                          weight_decay=0.001, 
                                          num_classes=2)
  load_weights_fun(wood_model_path, model_wood, optimizer_wood)

  model_rock, optimizer_rock, device = model_select(lr=0.001, 
                                            momentum=0.99, 
                                            weight_decay=0.001, 
                                          num_classes=2)
  load_weights_fun(rock_model_path, model_rock, optimizer_rock)

  models = {'wood' : model_wood, 'rock' : model_rock}

  print('start predicting boxes!')
  output_wood_list, output_rock_list, indx_list = validation_fun(models, data_loader, device)

  output_wood_list = [output['boxes'][output['scores'] > score_threshold] for output in output_wood_list]
  output_rock_list = [output['boxes'][output['scores'] > score_threshold] for output in output_rock_list]

  print('combine predicted boxes!')
  output_rock_list = combine_boxes(output_rock_list, device)

  return output_rock_list, output_wood_list


def calculate_sub(output_rock_list, output_wood_list, runs_dep_list, df2_test, device):

  runs_L10_list = calc_runs_length_all_boxes(output_rock_list, output_wood_list, device)


  rqs_list = calc_rq(runs_dep_list, runs_L10_list)

  last_list = []
  last_list.append(['RunId','Prediction'])
  for fl_nm, fl_rqs in zip(df2_test.File.to_list(), rqs_list):
    for i, rq_sngl_run in enumerate(fl_rqs, start=1):
      last_list.append([fl_nm.split('.')[0]+'-'+str(i), str(type_selection(rq_sngl_run))])

  sub = np.array(last_list)
  np.savetxt(r'output.csv', sub, fmt='%s', delimiter=',')

  return sub