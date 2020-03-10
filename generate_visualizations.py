import torch
from torchvision.transforms import transforms
import numpy as np
from tqdm import tqdm
import h5py
import os

import argparse

from modules.vgg_fg import vgg19 as vgg19_fg
from modules.vgg import vgg19 as vgg19_agf
from modules.vgg_rap import vgg19 as vgg19_rap
from torchvision.models import vgg19
from baselines.fullgrad import FullGrad

from modules.utils import *

from baselines.vanilla_backprop import VanillaBackprop
from baselines.smooth_grad import generate_smooth_grad
from baselines.integrated_gradients import IntegratedGradients
from baselines.gradcam import GradCam

from torchvision.datasets import ImageNet


def generate_visualization_to_hdf5(args):
    if args.method == 'fullgrad':
        fullgrad = FullGrad(model, device)
    first = True
    with h5py.File(os.path.join(args.method_dir, 'results.hdf5'), 'a') as f:
        data_cam = f.create_dataset('vis',
                                    (1, 1, 224, 224),
                                    maxshape=(None, 1, 224, 224),
                                    dtype=np.float32,
                                    compression="gzip")
        data_image = f.create_dataset('image',
                                      (1, 3, 224, 224),
                                      maxshape=(None, 3, 224, 224),
                                      dtype=np.float32,
                                      compression="gzip")
        data_target = f.create_dataset('target',
                                       (1,),
                                       maxshape=(None,),
                                       dtype=np.int32,
                                       compression="gzip")
        for batch_idx, (data, target) in enumerate(tqdm(sample_loader)):
            if first:
                first = False
                data_cam.resize(data_cam.shape[0] + data.shape[0] - 1, axis=0)
                data_image.resize(data_image.shape[0] + data.shape[0] - 1, axis=0)
                data_target.resize(data_target.shape[0] + data.shape[0] - 1, axis=0)
            else:
                data_cam.resize(data_cam.shape[0] + data.shape[0], axis=0)
                data_image.resize(data_image.shape[0] + data.shape[0], axis=0)
                data_target.resize(data_target.shape[0] + data.shape[0], axis=0)

            # Add data
            data_image[-data.shape[0]:] = data.data.cpu().numpy()
            data_target[-data.shape[0]:] = target.data.cpu().numpy()

            target = target.to(device)

            data = normalize(data)
            data = data.to(device)
            data.requires_grad_()

            if args.method == 'fullgrad':
                if args.vis_class == 'top':
                    cam = fullgrad.saliency(data)
                elif args.vis_class == 'index':
                    cam = fullgrad.saliency(data, target_class=[args.class_id] * len(target))
                elif args.vis_class == 'target':
                    target_list = target.tolist()
                    cam = fullgrad.saliency(data, target_class=target_list)
                else:
                    raise Exception('Invalid vis-class')
            elif 'agf' in args.method:
                model.zero_grad()
                model(data)

                kwargs = {
                    'no_a': args.no_ia,
                    'no_fx': args.no_fx,
                    'no_fdx': args.no_fgx,
                    'no_m': args.no_m,
                    'no_reg': args.no_reg,
                    'gradcam': args.gradcam
                }

                if args.vis_class == 'top':
                    cam = model.AGF(**kwargs)
                elif args.vis_class == 'index':
                    cam = model.AGF(class_id=[args.class_id] * len(target), **kwargs)
                elif args.vis_class == 'target':
                    target_list = target.tolist()
                    cam = model.AGF(class_id=target_list, **kwargs)
                else:
                    raise Exception('Invalid vis-class')
            elif args.method == 'rap':
                pred = model(data)

                T = clrp_target(pred, args.vis_class, target=target, class_id=args.class_id)
                cam = model.RAP_relprop(R=T)
                cam = cam.sum(dim=1, keepdim=True)
            elif args.method == 'lrp':
                pred = model(data)

                T = clrp_target(pred, args.vis_class, target=target, class_id=args.class_id)
                cam = model.relprop(R=T, alpha=1)
                cam = cam.sum(dim=1, keepdim=True)
            elif args.method == 'lrp_ab':
                pred = model(data)

                T = clrp_target(pred, args.vis_class, target=target, class_id=args.class_id)
                cam = model.relprop(R=T, alpha=2)
                cam = cam.sum(dim=1, keepdim=True)
            elif args.method == 'clrp':
                target = target.unsqueeze(1)
                pred = model(data)

                Tt = clrp_target(pred, args.vis_class, target=target, class_id=args.class_id)
                To = clrp_others(pred, args.vis_class, target=target, class_id=args.class_id)

                clrp_rel_target = model.relprop(R=Tt, alpha=1)
                clrp_rel_others = model.relprop(R=To, alpha=1)

                clrp_rscale = clrp_rel_target.sum(dim=[1, 2, 3], keepdim=True) / clrp_rel_others.sum(dim=[1, 2, 3],
                                                                                                     keepdim=True)
                clrp_rel = clrp_rel_target - clrp_rel_others * clrp_rscale
                cam = clrp_rel.sum(dim=1, keepdim=True)
            elif args.method == 'sglrp':
                target = target.unsqueeze(1)
                pred = model(data)

                Tt = sglrp_target(pred, args.vis_class, target=target, class_id=args.class_id)
                To = sglrp_others(pred, args.vis_class, target=target, class_id=args.class_id)

                sglrp_rel_target = model.relprop(R=Tt, alpha=1)
                sglrp_rel_others = model.relprop(R=To, alpha=1)

                sglrp_rscale = sglrp_rel_target.sum(dim=[1, 2, 3], keepdim=True) / sglrp_rel_others.sum(dim=[1, 2, 3],
                                                                                                        keepdim=True)
                sglrp_rel = sglrp_rel_target - sglrp_rel_others * sglrp_rscale
                cam = sglrp_rel.sum(dim=1, keepdim=True)
            elif args.method == 'smoothgrad':
                pred = model(data)

                if args.vis_class == 'top':
                    pred_class = pred.data.max(1, keepdim=True)[1].squeeze(1).item()
                elif args.vis_class == 'index':
                    pred_class = args.class_id
                elif args.vis_class == 'target':
                    pred_class = target.item()
                else:
                    raise Exception('Invalid vis-class')

                model.zero_grad()
                param_n = 50
                param_sigma_multiplier = 4
                cam = generate_smooth_grad(args.BP,
                                           data,
                                           pred_class,
                                           param_n,
                                           param_sigma_multiplier)
                cam = cam.abs().sum(dim=1, keepdim=True)
            elif args.method == 'integrad':
                pred = model(data)
                if args.vis_class == 'top':
                    pred_class = pred.data.max(1, keepdim=True)[1].squeeze(1).item()
                elif args.vis_class == 'index':
                    pred_class = args.class_id
                elif args.vis_class == 'target':
                    pred_class = target.item()
                else:
                    raise Exception('Invalid vis-class')

                cam = args.BP.generate_integrated_gradients(data,
                                                            pred_class,
                                                            100)
                cam = cam.abs().sum(dim=1, keepdim=True)
            elif args.method == 'gradcam':
                pred = model(data)
                if args.vis_class == 'top':
                    pred_class = pred.data.max(1, keepdim=True)[1].squeeze(1).item()
                elif args.vis_class == 'index':
                    pred_class = args.class_id
                elif args.vis_class == 'target':
                    pred_class = target.item()
                else:
                    raise Exception('Invalid vis-class')

                cam = args.BP.generate_cam(data,
                                           pred_class)
            else:
                raise Exception('No method found')

            data_cam[-data.shape[0]:] = cam.data.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a segmentation')
    parser.add_argument('--batch-size', type=int,
                        default=8,
                        help='Batch size')
    parser.add_argument('--method', type=str,
                        default='agf',
                        help='Method')
    parser.add_argument('--vis-class', type=str,
                        default='target',
                        choices=['top', 'target', 'index'],
                        help='Propagated class')
    parser.add_argument('--class-id', type=int,
                        default=0,
                        help='If --vis-class == index, then propagate --class-id')

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
    os.makedirs(os.path.join(PATH, 'visualizations'), exist_ok=True)

    os.makedirs(os.path.join(PATH, 'visualizations/{}'.format(args.method)), exist_ok=True)
    if args.vis_class == 'index':
        os.makedirs(os.path.join(PATH, 'visualizations/{}/{}_{}'.format(args.method,
                                                                        args.vis_class,
                                                                        args.class_id)), exist_ok=True)
        args.method_dir = os.path.join(PATH, 'visualizations/{}/{}_{}'.format(args.method,
                                                                              args.vis_class,
                                                                              args.class_id))
    else:
        os.makedirs(os.path.join(PATH, 'visualizations/{}/{}'.format(args.method,
                                                                     args.vis_class)), exist_ok=True)
        args.method_dir = os.path.join(PATH, 'visualizations/{}/{}'.format(args.method,
                                                                           args.vis_class))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # Load pre-trained models
    if args.method == 'fullgrad':
        model = vgg19_fg(pretrained=True).to(device)
        model.eval()
    elif 'agf' in args.method:
        model = vgg19_agf(pretrained=True).to(device)
        model.eval()
    elif args.method in ['rap', 'lrp', 'lrp_ab', 'clrp', 'sglrp']:
        if args.method == 'rap':
            args.batch_size = 1
        model = vgg19_rap(pretrained=True).to(device)
        model.eval()
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
        args.BP = GradCam(model, target_layer=36)   # True for VGG-19
    else:
        raise Exception('No model found')

    # Dataset loader for sample images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    imagenet_ds = ImageNet('path/to/imagenet/', split='val', download=False,
                           transform=transform)
    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    generate_visualization_to_hdf5(args)
