import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import data.transforms as transforms

from data.VOC import VOCClassification, VOCSBDClassification
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.parallel import DataParallelModel, DataParallelCriterion

from torch import nn
from torchvision.models import vgg19

import os
import argparse
from tqdm import tqdm

cudnn.benchmark = True

# Args
parser = argparse.ArgumentParser(description='Training multi-class classifier')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                    help='Learning rate')
parser.add_argument('--train-dataset', type=str, default='VOC', metavar='N',
                    help='Training Dataset')
parser.add_argument('--interval', type=int, default=10, metavar='N',
                    help='Evaluate every # of epochs')
parser.add_argument('--arc', type=str, default='vgg', metavar='N',
                    help='Model architecture')
# checking point
parser.add_argument('--resume', type=str, default=None,
                    help='put the path to resuming file if needed')
args = parser.parse_args()

args.checkname = args.arc

# Define Saver
saver = Saver(args)
saver.save_experiment_config()

# Define Tensorboard Summary
summary = TensorboardSummary(saver.experiment_dir)
writer = summary.create_summary()

# Data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_trans = transforms.Compose([transforms.Resize(321),
                                  transforms.RandomCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  normalize,
                                  ])
val_trans = transforms.Compose([transforms.Resize(321),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                                ])
train_ds = VOCSBDClassification('/path/to/VOC',
                                '/path/to/SBD/benchmark_RELEASE/dataset',
                                transform=train_trans, image_set='train')
train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
val_ds = VOCClassification('/path/to/VOC', transform=val_trans, image_set='val')
val_dl = DataLoader(val_ds, batch_size=8, shuffle=True, num_workers=2, drop_last=True)


# Model
if args.arc == 'vgg':
    model = vgg19(pretrained=True)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, train_ds.CLASSES)
    model = DataParallelModel(model.cuda())
else:
    raise Exception("Architecture {} not found".format(args.arc))

criterion = DataParallelCriterion(nn.BCEWithLogitsLoss().cuda())
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.2)
best_pred = 0

# Load model
if args.resume:
    if not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    model.module.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_pred = checkpoint['best_pred']

for epoch in range(args.epochs):
    model.train()
    scheduler.step()

    iterator = tqdm(train_dl)
    for batch_idx, (image, labels) in enumerate(iterator):
        image = image.cuda()
        labels = labels.cuda()

        predictions = model(image)

        loss = criterion(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iterator.set_description(
            'Epoch [{epoch}/{epochs}] :: Loss {loss:.4f}'.format(epoch=epoch + 1, epochs=args.epochs, loss=loss.item()))
        writer.add_scalar('train/loss', loss.item(), epoch * len(train_dl) + batch_idx)

    if epoch % args.interval == 0:
        model.eval()

        iterator = tqdm(val_dl)

        for batch_idx, (image, labels) in enumerate(iterator):
            image = image.cuda()
            labels = labels.cuda()

            predictions = model(image)
            loss = criterion(predictions, labels)

            # Mulit GPU handling
            if type(predictions) is list:
                predictions = torch.cat(predictions)
            pred_soft = torch.sigmoid(predictions)
            pred_binary = pred_soft.gt(0.5).type(predictions.type())

            iterator.set_description(
                'Val :: Loss {loss:.4f}'.format(loss=loss.item()))
            writer.add_scalar('val/loss', loss.item(), (epoch // args.interval) * len(val_dl) + batch_idx)

        saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        })
