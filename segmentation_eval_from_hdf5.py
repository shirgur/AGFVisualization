import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import glob

from modules.vgg import vgg19
from modules.layers import Linear
from utils.metrices import *
from data.VOC import VOCResults

import os
import argparse
from tqdm import tqdm


def eval(args):
    ap_meter = AverageMeter()
    acc_meter = AverageMeter()

    iterator = tqdm(sample_loader)

    for batch_idx, (_, vis, target, class_pred) in enumerate(iterator):
        vis = vis.to(device)
        target = target.to(device)

        binary_pred = torch.sigmoid(class_pred).gt(0.5).long()

        predictions = vis.clone()
        predictions = predictions.clamp(min=0)
        predictions[:, 1:] += -0.1 * (1-binary_pred[:, :, None, None].cuda().float())
        predictions = F.softmax(predictions, 1)

        # Average precision
        for i in range(target.shape[0]):
            ap = get_ap_scores(predictions[i:i+1], target[i:i+1])[0]
            acc, pix = pix_accuracy(predictions[i:i+1], target[i:i+1])
            acc_meter.update(acc, pix)
            ap_meter.update(ap)

        iterator.set_description('pixAcc: %.4f, mAP: %.4f' % (acc_meter.average() * 100, ap_meter.average()))

    txtfile = os.path.join(args.experiment_dir, 'results.txt')
    with open(txtfile, 'w') as f:
        print('[Eval Summary]:')
        print('Accuracy: {:.2f}%, AP: {:.2f}'.format(acc_meter.average() * 100, ap_meter.average() * 100))

        f.write('[Eval Summary]:')
        f.write('Accuracy: {:.2f}%, AP: {:.2f}'.format(acc_meter.average() * 100, ap_meter.average() * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a segmentation')
    parser.add_argument('--batch-size', type=int,
                        default=8,
                        help='Batch size')
    parser.add_argument('--method', type=str,
                        default='agf',
                        help='Method')
    parser.add_argument('--resume', type=str,
                        default='/path/to/run/VOC/vgg/experiment_0/checkpoint.pth.tar',
                        help='Saved model path')
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    # PATH variables
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    dataset = PATH + 'dataset/'
    os.makedirs(os.path.join(PATH, 'experiments'), exist_ok=True)
    os.makedirs(os.path.join(PATH, 'experiments/segmentation'), exist_ok=True)

    exp_name = args.method
    args.runs_dir = os.path.join(PATH, 'experiments/segmentation/{}'.format(exp_name))
    experiments = sorted(glob.glob(os.path.join(args.runs_dir, 'experiment_*')))
    experiment_id = int(experiments[-1].split('_')[-1]) + 1 if experiments else 0
    args.experiment_dir = os.path.join(args.runs_dir, 'experiment_{}'.format(str(experiment_id)))
    os.makedirs(args.experiment_dir, exist_ok=True)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    vis_method_dir = 'visualizations_multiclass/{}'.format(args.method)

    imagenet_ds = VOCResults(vis_method_dir)

    # Model
    seg_classes = imagenet_ds.CLASSES + 1
    kwargs = {'num_classes': imagenet_ds.CLASSES}
    model = vgg19(pretrained=True)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = Linear(num_ftrs, imagenet_ds.CLASSES)
    model = model.cuda()

    # Load model
    if not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    save_path = PATH + 'results/'

    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=args.batch_size,
        shuffle=False)

    eval(args)
