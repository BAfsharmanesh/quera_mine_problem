import math
import sys
import os
from typing import Tuple, List, Dict, Optional
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import torch
from torch import Tensor
from torch.utils import data
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers

from src.dataset import Dataset_from_memory, RandomHorizontalFlip, img_transform
from src.model import model_select
from src.utils import collate_fn


def train_one_epoch(model, 
                    optimizer, 
                    data_loader, 
                    device, epoch, 
                    log_interval, 
                    verbose):
    model.train()
    step_loss = []
    N_count = 0
    n_total_steps = len(data_loader.dataset)
    
    for batch_idx, (X, y) in enumerate(data_loader):
        X = list(img.to(device) for img in X)
        y = [{k: v.to(device) for k, v in t.items()} for t in y]

        N_count += len(X)

        loss_dict = model(X, y)

        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        step_loss.append(loss_value)
        # show information
        if (batch_idx + 1) % log_interval == 0:
            
            if verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                    epoch + 1, 
                    N_count, 
                    n_total_steps, 
                    100. * (batch_idx + 1) / len(data_loader), 
                    sum(step_loss)/(batch_idx + 1)
                    ))
            
        
    train_loss = sum(step_loss)/len(data_loader)
    
    if verbose:
        print('\nTrain Epoch: {} : Average loss: {:.6f}'.format(epoch + 1, train_loss))

    return train_loss


def valid_one_epoch(model, 
                    data_loader, 
                    device, 
                    epoch, 
                    verbose):
    model.eval()
    step_loss = []
    N_count = 0
    n_total_steps = len(data_loader.dataset)
    
    for batch_idx, (X, y) in enumerate(data_loader):
        X = list(img.to(device) for img in X)
        y = [{k: v.to(device) for k, v in t.items()} for t in y]

        N_count += len(X)

        loss_dict, detections = eval_forward(model,X, y)

        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping validating".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        step_loss.append(loss_value)
            
    valid_loss = sum(step_loss)/len(data_loader)
    
    if verbose:
        print('Valid Epoch: {} : Average loss: {:.6f}\n'.format(epoch + 1, valid_loss))

    return valid_loss


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                print(f'early stopper is activated after {self.counter} steps without improvment')
                return True
        return False


def training_plot(trainingEpoch_loss, validationEpoch_loss):
  plt.plot(trainingEpoch_loss, label='train_loss')
  plt.plot(validationEpoch_loss,label='val_loss')
  plt.legend()
  plt.xlabel("epoch")
  plt.ylabel("loss")
  plt.show()


def train_valid_setup(lr, 
                        momentum, 
                        weight_decay, 
                        batch_size, 
                        num_epochs, 
                        train_indx, 
                        valid_indx, 
                        all_img_tensor, 
                        targets_dict_list, 
                        num_classes,
                        modelType):
    
    modelTypeName = modelType['name']

    conf_params = {'drop_last':True, 'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True, 'collate_fn':collate_fn}
    train_set =   Dataset_from_memory([targets_dict_list[i] for i in train_indx], all_img_tensor[train_indx,:,:,:],
                                        transform=img_transform() if modelType['imgT'] else None,
                                        imgBoxTransform=RandomHorizontalFlip(0.5),
                                        train=True)
    train_loader = data.DataLoader(train_set, **conf_params)
    valid_set =   Dataset_from_memory([targets_dict_list[i] for i in valid_indx], all_img_tensor[valid_indx,:,:,:], 
                                        transform=img_transform() if modelType['imgT'] else None,
                                        imgBoxTransform=None,
                                        train=True)
    valid_loader = data.DataLoader(valid_set, **conf_params)

    model, optimizer, device = model_select(lr, momentum, weight_decay, num_classes)

    trainingEpoch_loss = []
    validationEpoch_loss = []
    early_stopper = EarlyStopper(patience=modelType['patience'], min_delta=0.01)

    for epoch in tqdm(range(num_epochs)):
        # train, test model
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, log_interval=10, verbose=True)
        valid_loss = valid_one_epoch(model, valid_loader, device, epoch, verbose=True)
        trainingEpoch_loss.append(train_loss)
        validationEpoch_loss.append(valid_loss)
        
        if (valid_loss < early_stopper.min_validation_loss) and (valid_loss<=modelType['max']):
            torch.save({
                        'epoch': epoch+1,
                        'train_loss': train_loss,
                        'valid_loss': valid_loss,
                        'R_CNN': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        }, 'models/'+os.path.join(f'checkPoint_{modelTypeName}_model.pth'))
            print(f'model of epoch {epoch+1} saved')
        
        if early_stopper.early_stop(valid_loss):             
            break

    return trainingEpoch_loss, validationEpoch_loss




























def eval_forward(model, images, targets):
    """ type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]] """
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            It returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).
    """
    model.eval()

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    model.rpn.training=True
    #model.roi_heads.training=True


    #####proposals, proposal_losses = model.rpn(images, features, targets)
    features_rpn = list(features.values())
    objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
    anchors = model.rpn.anchor_generator(images, features_rpn)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    proposals, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    proposal_losses = {}
    assert targets is not None
    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
    regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    proposal_losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }

    #####detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    image_shapes = images.image_sizes
    proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}
    loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
    num_images = len(boxes)
    for i in range(num_images):
        result.append(
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )
    detections = result
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
    model.rpn.training=False
    model.roi_heads.training=False
    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses, detections