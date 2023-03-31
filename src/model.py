import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, faster_rcnn
from torchvision.models import ResNet50_Weights

from src.utils import seed_everything


def model_select(lr, momentum, weight_decay, num_classes):
    seed_everything(seed=42)
    # Detect devices
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
    print(device)

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, 
                                    progress=False, 
                                    weights_backbone=ResNet50_Weights.DEFAULT,
                                    trainable_backbone_layers=None).to(device)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes).to(device)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr,
                                momentum=momentum, weight_decay=weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                 patience=patience,
    #                                                 factor=factor,
    #                                                 verbose=True)
    
    return model, optimizer, device