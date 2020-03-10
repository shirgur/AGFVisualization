import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from modules.layers_rap import *

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = AdaptiveAvgPool2d((7, 7))
        self.classifier = Sequential(
            Linear(512 * 7 * 7, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def relprop(self, R, alpha):
        x = self.classifier.relprop(R, alpha)
        x = x.reshape_as(next(reversed(self.features._modules.values())).Y)
        x = self.avgpool.relprop(x, alpha)
        x = self.features.relprop(x, alpha)

        return x

    def m_relprop(self, R, pred, alpha):
        x = self.classifier.m_relprop(R, pred, alpha)
        if torch.is_tensor(x) == False:
            for i in range(len(x)):
                x[i] = x[i].reshape_as(next(reversed(self.features._modules.values())).Y)
        else:
            x = x.reshape_as(next(reversed(self.features._modules.values())).Y)
        x = self.avgpool.m_relprop(x, pred, alpha)
        x = self.features.m_relprop(x, pred, alpha)

        return x

    def RAP_relprop(self, R):
        x1 = self.classifier.RAP_relprop(R)
        if torch.is_tensor(x1) == False:
            for i in range(len(x1)):
                x1[i] = x1[i].reshape_as(next(reversed(self.features._modules.values())).Y)
        else:
            x1 = x1.reshape_as(next(reversed(self.features._modules.values())).Y)
        x1 = self.avgpool.RAP_relprop(x1)
        x1 = self.features.RAP_relprop(x1)

        return x1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, BatchNorm2d(v), ReLU(inplace=True)]
            else:
                layers += [conv2d, ReLU(inplace=True)]
            in_channels = v
    return Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model


if __name__ == '__main__':
    import numpy as np
    import imageio
    import cv2
    from utils import render


    def normalize(tensor,
                  mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]):
        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        return tensor


    def compute_pred(output):
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        print('Pred cls : ' + str(pred))
        T = pred.squeeze().cpu().numpy()
        T = np.expand_dims(T, 0)
        T = (T[:, np.newaxis] == np.arange(1000)) * 1.0
        T = torch.from_numpy(T).type(torch.FloatTensor)
        Tt = T.cuda()

        return Tt

    # CLRP
    def clrp_target(output):
        pred = output.data.max(1, keepdim=True)[1]
        # pred[0, 0] = 282
        T = torch.zeros_like(output)
        T.scatter_(1, pred, 1)
        return T * output


    def clrp_others(output):
        pred = output.data.max(1, keepdim=True)[1]
        # pred[0, 0] = 282
        T = torch.ones_like(output)
        T.scatter_(1, pred, 0)
        T /= (output.shape[-1] - 1)
        return T * output.gather(1, pred)

    # SGLRP
    def sglrp_target(output):
        pred = output.data.max(1, keepdim=True)[1]
        # pred[0, 0] = 282
        sm_pred = torch.softmax(output, dim=1)
        T = torch.zeros_like(output)
        T.scatter_(1, pred, 1)
        return T * (sm_pred * (1 - sm_pred) + 1e-8)


    def sglrp_others(output):
        pred = output.data.max(1, keepdim=True)[1]
        # pred[0, 0] = 282
        sm_pred = torch.softmax(output, dim=1)
        T = torch.ones_like(output)
        T.scatter_(1, pred, 0)
        return T * sm_pred.gather(1, pred) * sm_pred


    model = vgg19(pretrained=True).cuda()
    model.eval()
    image0 = imageio.imread("../dataset/imagenet/ILSVRC2012_val_00000000.png")
    image1 = imageio.imread("../dataset/imagenet/ILSVRC2012_val_00000001.JPEG")
    image0 = cv2.resize(image0, (224, 224)) / 255
    image1 = cv2.resize(image1, (224, 224)) / 255
    image0 = torch.tensor(image0).permute(2, 0, 1).unsqueeze(0).float().cuda()
    image1 = torch.tensor(image1).permute(2, 0, 1).unsqueeze(0).float().cuda()
    image = torch.cat((image0, image1), dim=0)
    image = normalize(image)
    image.requires_grad_()

    output = model(image)

    # LRP
    Tt = clrp_target(output)

    lrp_rel = model.relprop(R=Tt, alpha=1)

    clrp_maps = (render.hm_to_rgb(lrp_rel[0, 0].data.cpu().numpy(), scaling=3, sigma=1, cmap='seismic') * 255).astype(np.uint8)
    imageio.imsave('../lrp_hm0.jpg', clrp_maps)
    clrp_maps = (render.hm_to_rgb(lrp_rel[1, 0].data.cpu().numpy(), scaling=3, sigma=1, cmap='seismic') * 255).astype(np.uint8)
    imageio.imsave('../lrp_hm1.jpg', clrp_maps)

    # CLRP
    Tt = clrp_target(output)
    To = clrp_others(output)

    clrp_rel_target = model.relprop(R=Tt, alpha=1)
    clrp_rel_others = model.relprop(R=To, alpha=1)

    clrp_rscale = clrp_rel_target.sum(dim=[1, 2, 3], keepdim=True) / clrp_rel_others.sum(dim=[1, 2, 3], keepdim=True)
    clrp_rel = clrp_rel_target - clrp_rel_others * clrp_rscale

    clrp_maps = (render.hm_to_rgb(clrp_rel[0, 0].data.cpu().numpy(), scaling=3, sigma=1, cmap='seismic') * 255).astype(np.uint8)
    imageio.imsave('../clrp_hm0.jpg', clrp_maps)
    clrp_maps = (render.hm_to_rgb(clrp_rel[1, 0].data.cpu().numpy(), scaling=3, sigma=1, cmap='seismic') * 255).astype(np.uint8)
    imageio.imsave('../clrp_hm1.jpg', clrp_maps)

    # SGLRP
    Tt = sglrp_target(output)
    To = sglrp_others(output)

    sglrp_rel_target = model.relprop(R=Tt, alpha=1)
    sglrp_rel_others = model.relprop(R=To, alpha=1)

    sglrp_rscale = sglrp_rel_target.sum(dim=[1, 2, 3], keepdim=True) / sglrp_rel_others.sum(dim=[1, 2, 3], keepdim=True)
    sglrp_rel = sglrp_rel_target - sglrp_rel_others * sglrp_rscale

    sglrp_maps = (render.hm_to_rgb(sglrp_rel[0, 0].data.cpu().numpy(), scaling=3, sigma=1, cmap='seismic') * 255).astype(np.uint8)
    imageio.imsave('../sglrp_hm0.jpg', sglrp_maps)
    sglrp_maps = (render.hm_to_rgb(sglrp_rel[1, 0].data.cpu().numpy(), scaling=3, sigma=1, cmap='seismic') * 255).astype(np.uint8)
    imageio.imsave('../sglrp_hm1.jpg', sglrp_maps)


