import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import h5py

from torch import nn
from torchvision.models import vgg19

from modules.vgg_fg import vgg19 as vgg19_fg
import modules.layers as nn_new
from modules.vgg import vgg19 as vgg19_agf
import modules.layers_rap as nn_rap
from modules.vgg_rap import vgg19 as vgg19_rap
from modules.utils import *

from baselines.vanilla_backprop import VanillaBackprop
from baselines.smooth_grad import generate_smooth_grad
from baselines.integrated_gradients import IntegratedGradients
from baselines.gradcam import GradCam
from baselines.fullgrad import FullGrad

from data.VOC import VOCSegmentation

import os
import argparse
from tqdm import tqdm
import numpy as np


cudnn.benchmark = True


parser = argparse.ArgumentParser(description='Training multi-class classifier')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='Batch size')
parser.add_argument('--train-dataset', type=str, default='VOC', metavar='N',
                    help='Training Dataset')
parser.add_argument('--method', type=str,
                    default='agf',
                    help='Method')
# checking point
parser.add_argument('--resume', type=str,
                    default='/path/to/run/VOC/vgg/experiment_0/checkpoint.pth.tar',
                    help='Path to trained weights')
args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
os.makedirs(os.path.join(PATH, 'visualizations_multiclass'), exist_ok=True)
os.makedirs(os.path.join(PATH, 'visualizations_multiclass/{}'.format(args.method)), exist_ok=True)
args.method_dir = os.path.join(PATH, 'visualizations_multiclass/{}'.format(args.method))

# Data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

test_img_trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])
test_lbl_trans = transforms.Compose([
    transforms.Resize((224, 224), Image.NEAREST)
])

test_ds = VOCSegmentation('/path/to/VOC',
                          transform=test_img_trans, target_transform=test_lbl_trans, image_set='val')
test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=1)

# Model
seg_classes = test_ds.CLASSES + 1
if args.method == 'gradcam':
    model = vgg19(pretrained=False)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, test_ds.CLASSES)
    model = model.cuda()
    BP = GradCam(model, target_layer=36)
elif args.method == 'fullgrad':
    model = vgg19_fg(pretrained=False)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, test_ds.CLASSES)
    model = model.cuda()
    fullgrad = FullGrad(model, device)
elif 'agf' in args.method:
    model = vgg19_agf(pretrained=False)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn_new.Linear(num_ftrs, test_ds.CLASSES)
    model = model.cuda()
elif args.method in ['rap', 'lrp', 'lrp_ab', 'clrp', 'sglrp']:
    model = vgg19_rap(pretrained=False)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn_rap.Linear(num_ftrs, test_ds.CLASSES)
    model = model.cuda()
elif args.method == 'integrad':
    model = vgg19(pretrained=False)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, test_ds.CLASSES)
    model = model.cuda()
    BP = IntegratedGradients(model)
elif args.method == 'smoothgrad':
    model = vgg19(pretrained=False)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, test_ds.CLASSES)
    model = model.cuda()
    BP = VanillaBackprop(model)
else:
    raise Exception('No method found')

# Load model
if not os.path.isfile(args.resume):
    raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
checkpoint = torch.load(args.resume)
model.load_state_dict(checkpoint['state_dict'])

model.eval()

iterator = tqdm(test_dl)

with h5py.File(os.path.join(args.method_dir, 'results.hdf5'), 'a') as f:
    data_cam = f.create_dataset('vis',
                                (1, seg_classes, 224, 224),
                                maxshape=(None, seg_classes, 224, 224),
                                dtype=np.float32,
                                compression="gzip")
    data_image = f.create_dataset('image',
                                  (1, 3, 224, 224),
                                  maxshape=(None, 3, 224, 224),
                                  dtype=np.float32,
                                  compression="gzip")
    data_target = f.create_dataset('target',
                                   (1, 224, 224),
                                   maxshape=(None, 224, 224),
                                   dtype=np.int32,
                                   compression="gzip")
    data_class_pred = f.create_dataset('class_pred',
                                       (1, seg_classes - 1),
                                       maxshape=(None, seg_classes - 1),
                                       dtype=np.float32,
                                       compression="gzip")
    first = True
    for batch_idx, (image, labels) in enumerate(iterator):
        if first:
            first = False
            data_cam.resize(data_cam.shape[0] + image.shape[0] - 1, axis=0)
            data_image.resize(data_image.shape[0] + image.shape[0] - 1, axis=0)
            data_target.resize(data_target.shape[0] + image.shape[0] - 1, axis=0)
            data_class_pred.resize(data_target.shape[0] + image.shape[0] - 1, axis=0)
        else:
            data_cam.resize(data_cam.shape[0] + image.shape[0], axis=0)
            data_image.resize(data_image.shape[0] + image.shape[0], axis=0)
            data_target.resize(data_target.shape[0] + image.shape[0], axis=0)
            data_class_pred.resize(data_target.shape[0] + image.shape[0], axis=0)

        image = image.cuda()
        image.requires_grad_()
        labels = labels.cuda()

        predictions = model(image)
        binary_pred = torch.sigmoid(predictions).gt(0.5).type(predictions.type())

        output = torch.zeros(image.shape[0], seg_classes, *image.shape[2:]).to(image.device)

        for i, single_output in enumerate(output):
            num_classes = int(binary_pred[i].sum().item())

            if num_classes > 0:
                _, topk_idx = torch.topk(predictions[i], num_classes, dim=-1)
                topk_idx = topk_idx.tolist()

                for class_id in topk_idx:
                    if 'agf' in args.method:
                        model.zero_grad()
                        model(image)
                        cam = model.AGF(class_id=[class_id])
                        cam = cam / cam.max()
                        cam_sum = cam.sum().item()
                    elif args.method == 'gradcam':
                        model.zero_grad()
                        model(image)
                        cam = BP.generate_cam(image, class_id)
                        cam = cam - cam.mean()
                        cam = cam / cam.max()
                    elif args.method == 'integrad':
                        model.zero_grad()
                        model(image)
                        cam = BP.generate_integrated_gradients(image, class_id, 100)
                        cam = cam - cam.mean()
                        cam = cam / cam.max()
                    elif args.method == 'smoothgrad':
                        model.zero_grad()
                        model(image)
                        param_n = 50
                        param_sigma_multiplier = 4
                        cam = generate_smooth_grad(BP,
                                                   image,
                                                   class_id,
                                                   param_n,
                                                   param_sigma_multiplier)
                        cam = cam - cam.mean()
                        cam = cam / cam.max()
                    elif args.method == 'fullgrad':
                        model.zero_grad()
                        model(image)
                        cam = fullgrad.saliency(image, target_class=[class_id])
                        cam = cam - cam.mean()
                        cam = cam / cam.max()
                    elif args.method == 'rap':
                        model.zero_grad()
                        pred = model(image)
                        T = clrp_target(pred, 'target', class_id=class_id)
                        cam = model.RAP_relprop(R=T)
                        cam = cam.sum(dim=1, keepdim=True)
                        cam = cam / cam.max()
                    elif args.method == 'lrp':
                        model.zero_grad()
                        pred = model(image)
                        T = clrp_target(pred, 'target', class_id=class_id)
                        cam = model.relprop(R=T, alpha=1)
                        cam = cam.sum(dim=1, keepdim=True)
                        cam = cam - cam.mean()
                        cam = cam / cam.max()
                    elif args.method == 'lrp_ab':
                        model.zero_grad()
                        pred = model(image)
                        T = clrp_target(pred, 'target', class_id=class_id)
                        cam = model.relprop(R=T, alpha=2)
                        cam = cam.sum(dim=1, keepdim=True)
                    elif args.method == 'clrp':
                        model.zero_grad()
                        pred = model(image)

                        Tt = clrp_target(pred, 'target', class_id=class_id)
                        To = clrp_others(pred, 'target', class_id=class_id)

                        clrp_rel_target = model.relprop(R=Tt, alpha=1)
                        clrp_rel_others = model.relprop(R=To, alpha=1)

                        clrp_rscale = clrp_rel_target.sum(dim=[1, 2, 3], keepdim=True) / clrp_rel_others.sum(
                            dim=[1, 2, 3], keepdim=True)
                        clrp_rel = clrp_rel_target - clrp_rel_others * clrp_rscale

                        cam = clrp_rel.sum(dim=1, keepdim=True)
                        cam = cam / cam.max()
                    elif args.method == 'sglrp':
                        model.zero_grad()
                        pred = model(image)

                        Tt = sglrp_target(pred, 'target', class_id=class_id)
                        To = sglrp_others(pred, 'target', class_id=class_id)

                        sglrp_rel_target = model.relprop(R=Tt, alpha=1)
                        sglrp_rel_others = model.relprop(R=To, alpha=1)

                        sglrp_rscale = sglrp_rel_target.sum(dim=[1, 2, 3], keepdim=True) / sglrp_rel_others.sum(
                            dim=[1, 2, 3], keepdim=True)
                        sglrp_rel = sglrp_rel_target - sglrp_rel_others * sglrp_rscale

                        cam = sglrp_rel.sum(dim=1, keepdim=True)
                        cam = cam / cam.max()
                    else:
                        raise Exception('No method found')

                    cam = cam.sum(dim=1)
                    output.data[i:i + 1, class_id + 1] = cam.data[i:i + 1].clone()

        data_cam[-image.shape[0]:] = output.data.cpu().numpy()
        data_image[-image.shape[0]:] = image.data.cpu().numpy()
        data_target[-image.shape[0]:] = labels.data.cpu().numpy()
        data_class_pred[-image.shape[0]:] = predictions.data.cpu().numpy()
