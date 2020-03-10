import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import argparse
from PIL import Image
import os
from tqdm import tqdm
import glob

from modules.vgg_fg import vgg19 as vgg19_fg
from modules.vgg import vgg19 as vgg19_agf
from modules.vgg_rap import vgg19 as vgg19_rap
from torchvision.models import vgg19

from modules.utils import *
from utils.metrices import *
from data.imagenet import Imagenet_Segmentation

from baselines.fullgrad import FullGrad
from baselines.vanilla_backprop import VanillaBackprop
from baselines.smooth_grad import generate_smooth_grad
from baselines.integrated_gradients import IntegratedGradients
from baselines.gradcam import GradCam

# Args
parser = argparse.ArgumentParser(description='Training multi-class classifier')
parser.add_argument('--arc', type=str, default='vgg', metavar='N',
                    help='Model architecture')
parser.add_argument('--train_dataset', type=str, default='imagenet', metavar='N',
                    help='Testing Dataset')
parser.add_argument('--method', type=str,
                    default='agf',
                    help='')
parser.add_argument('--thr', type=float, default=0.,
                    help='threshold')

# Ablation option from the our method
parser.add_argument('--no-a', action='store_true',
                    default=False,
                    help='No A')
parser.add_argument('--no-fx', action='store_true',
                    default=False,
                    help='No F_x')
parser.add_argument('--no-fdx', action='store_true',
                    default=False,
                    help='No F_dx')
parser.add_argument('--no-m', action='store_true',
                    default=False,
                    help='No M')
parser.add_argument('--no-reg', action='store_true',
                    default=False,
                    help='No regulatization by C')
parser.add_argument('--gradcam', action='store_true',
                    default=False,
                    help='Use GradCAM method as residual')
args = parser.parse_args()

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
os.makedirs(os.path.join(PATH, 'experiments'), exist_ok=True)
os.makedirs(os.path.join(PATH, 'experiments/imagenet_segmentation'), exist_ok=True)

args.runs_dir = os.path.join(PATH, 'experiments/perturbations/{}'.format(args.method))

experiments = sorted(glob.glob(os.path.join(args.runs_dir, 'experiment_*')))
experiment_id = int(experiments[-1].split('_')[-1]) + 1 if experiments else 0
args.experiment_dir = os.path.join(args.runs_dir, 'experiment_{}'.format(str(experiment_id)))
os.makedirs(args.experiment_dir, exist_ok=True)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
test_img_trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])
test_lbl_trans = transforms.Compose([
    transforms.Resize((224, 224), Image.NEAREST),
])

ds = Imagenet_Segmentation('/path/to/imagenet-seg/other/gtsegs_ijcv.mat',
                           transform=test_img_trans, target_transform=test_lbl_trans)
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

# Model
if args.arc == 'vgg':
    if 'agf' in args.method:
        model = vgg19_agf(pretrained=True).cuda()
    elif args.method in ['rap', 'lrp', 'lrp_ab', 'clrp', 'sglrp']:
        if args.method == 'rap':
            args.batch_size = 1
        model = vgg19_rap(pretrained=True).to(device)
        model.eval()
    elif args.method == 'fullgrad':
        model = vgg19_fg(pretrained=True).cuda()
        fullgrad = FullGrad(model, device)
    elif args.method == 'smoothgrad':
        args.batch_size = 1
        model = vgg19(pretrained=True).to(device)
        args.BP = VanillaBackprop(model)
    elif args.method == 'integrad':
        args.batch_size = 1
        model = vgg19(pretrained=True).to(device)
        args.BP = IntegratedGradients(model)
    elif args.method == 'gradcam':
        args.batch_size = 1
        model = vgg19(pretrained=True).to(device)
        args.BP = GradCam(model, target_layer=36)
    else:
        raise Exception("Type [{}] not found".format(args.method))
else:
    raise Exception("Architecture {} not found".format(args.arc))

model.eval()

iterator = tqdm(dl)


def eval_batch(image, labels, evaluator):
    evaluator.zero_grad()
    image.requires_grad = True

    image = image.requires_grad_()
    predictions = evaluator(image)

    if 'agf' in args.method:
        kwargs = {
            'no_a': args.no_ia,
            'no_fx': args.no_fx,
            'no_fdx': args.no_fgx,
            'no_m': args.no_m,
            'no_reg': args.no_reg,
            'gradcam': args.gradcam
        }
        Res = evaluator.AGF(**kwargs)
    elif args.method == 'rap':
        T = clrp_target(predictions, 'top')
        Res = model.RAP_relprop(R=T)
    elif args.method == 'fullgrad':
        Res = fullgrad.saliency(image)
        Res = Res - Res.mean()
    elif args.method == 'lrp':
        T = clrp_target(predictions, 'top')
        Res = model.relprop(R=T, alpha=1)
        Res = Res - Res.mean()
    elif args.method == 'lrp_ab':
        T = clrp_target(predictions, 'top')
        Res = model.relprop(R=T, alpha=2)
    elif args.method == 'clrp':
        Tt = clrp_target(predictions, 'top')
        To = clrp_others(predictions, 'top')

        clrp_rel_target = model.relprop(R=Tt, alpha=1)
        clrp_rel_others = model.relprop(R=To, alpha=1)

        clrp_rscale = clrp_rel_target.sum(dim=[1, 2, 3], keepdim=True) / clrp_rel_others.sum(dim=[1, 2, 3],
                                                                                             keepdim=True)
        Res = clrp_rel_target - clrp_rel_others * clrp_rscale
    elif args.method == 'sglrp':
        Tt = sglrp_target(predictions, 'top')
        To = sglrp_others(predictions, 'top')

        sglrp_rel_target = model.relprop(R=Tt, alpha=1)
        sglrp_rel_others = model.relprop(R=To, alpha=1)

        sglrp_rscale = sglrp_rel_target.sum(dim=[1, 2, 3], keepdim=True) / sglrp_rel_others.sum(dim=[1, 2, 3],
                                                                                                keepdim=True)
        Res = sglrp_rel_target - sglrp_rel_others * sglrp_rscale
    elif args.method == 'smoothgrad':
        pred_class = predictions.data.max(1, keepdim=True)[1].squeeze(1).item()

        model.zero_grad()
        param_n = 50
        param_sigma_multiplier = 4
        cam = generate_smooth_grad(args.BP,
                                   image,
                                   pred_class,
                                   param_n,
                                   param_sigma_multiplier)
        Res = cam.abs()
        Res = Res - Res.mean()
    elif args.method == 'integrad':
        pred_class = predictions.data.max(1, keepdim=True)[1].squeeze(1).item()

        cam = args.BP.generate_integrated_gradients(image,
                                                    pred_class,
                                                    100)
        Res = cam.abs()
        Res = Res - Res.mean()
    elif args.method == 'gradcam':
        pred_class = predictions.data.max(1, keepdim=True)[1].squeeze(1).item()

        cam = args.BP.generate_cam(image,
                                   pred_class)
        Res = cam.abs()
        Res = Res - Res.mean()
    else:
        raise Exception("Type [{}] not found".format(args.method))

    Res = Res.sum(dim=1, keepdim=True)
    Res_1 = Res.gt(args.thr).type(Res.type())
    Res_0 = Res.le(args.thr).type(Res.type())

    # TEST
    pred = Res.clamp(min=args.thr) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()

    output = torch.cat((Res_0, Res_1), 1)

    # Evaluate Segmentation
    batch_correct, batch_label = 0, 0
    batch_ap, = 0

    # Segmentation resutls
    correct, labeled = pix_accuracy(output[0].data.cpu(), labels[0])
    batch_correct += correct
    batch_label += labeled
    ap = np.nan_to_num(get_ap_scores(output, labels))
    batch_ap += ap

    return batch_correct, batch_label, batch_ap, pred, target


total_correct, total_label = np.int64(0), np.int64(0)
total_ap = []

predictions, targets = [], []
for batch_idx, (image, labels) in enumerate(iterator):
    images = image.cuda()
    labels = labels.cuda()

    correct, labeled, ap, pred, target = eval_batch(images, labels, model)

    predictions.append(pred)
    targets.append(target)

    total_correct += correct.astype('int64')
    total_label += labeled.astype('int64')
    total_ap += [ap]
    pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
    mAp = np.mean(total_ap)
    iterator.set_description('pixAcc: %.4f, mAP: %.4f' % (pixAcc, mAp))

txtfile = os.path.join(args.experiment_dir, 'result.txt')
fh = open(txtfile, 'w')
print("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
print("Mean AP over %d classes: %.4f\n" % (2, mAp))

fh.write("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
fh.write("Mean AP over %d classes: %.4f\n" % (2, mAp))
fh.close()
