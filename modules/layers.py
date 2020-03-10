import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['forward_hook', 'ReLU', 'Dropout', 'BatchNorm2d', 'Linear', 'MaxPool2d',
           'AdaptiveAvgPool2d', 'AvgPool2d', 'Conv2d', 'Sequential', 'safe_divide',
           'Add', 'Clone', 'minmax_dims']


# Misc
def get_factorization(X, grad, phi):
    # Phi partition
    phi_pos = phi.clamp(min=0)
    phi_neg = phi.clamp(max=0)

    # Normalize inputs
    grad = safe_divide(grad, minmax_dims(grad, 'max'))
    X = safe_divide(X, minmax_dims(X, 'max'))

    # Compute F_dx
    # Heaviside function
    ref = torch.sigmoid(10 * grad)

    # Compute representatives - R
    R_pos = safe_divide((ref * phi_pos.ne(0).type(ref.type())).sum(dim=[2, 3], keepdim=True), (
        phi_pos.ne(0).type(phi_pos.type()).sum(dim=[1, 2, 3], keepdim=True)))
    R_neg = safe_divide((ref * phi_neg.ne(0).type(ref.type())).sum(dim=[2, 3], keepdim=True), (
        phi_neg.ne(0).type(phi_neg.type()).sum(dim=[1, 2, 3], keepdim=True)))
    R = torch.cat((R_neg.squeeze(3), R_pos.squeeze(3)), dim=-1)

    # Compute weights - W
    H = ref.reshape(X.shape[0], X.shape[1], -1)
    W = ((R.permute(0, 2, 1) @ R + torch.eye(2)[None, :, :].cuda() * 1) * 1).inverse() @ (R.permute(0, 2, 1) @ H)
    W = F.relu(W.reshape(W.shape[0], W.shape[1], X.shape[2], X.shape[3]))
    F_dx = -(W[:, 0:1] - W[:, 1:2])
    F_dx = F_dx.expand_as(X)

    # Compute F_x
    # Heaviside function
    ref = torch.sigmoid(10 * X)

    # Compute representatives - R
    R_pos = safe_divide((ref * phi_pos.ne(0).type(ref.type())).sum(dim=[2, 3], keepdim=True), (
        phi_pos.ne(0).type(phi_pos.type()).sum(dim=[1, 2, 3], keepdim=True)))
    R_neg = safe_divide((ref * phi_neg.ne(0).type(ref.type())).sum(dim=[2, 3], keepdim=True), (
        phi_neg.ne(0).type(phi_neg.type()).sum(dim=[1, 2, 3], keepdim=True)))

    # Compute weights - W
    R = torch.cat((R_neg.squeeze(3), R_pos.squeeze(3)), dim=-1)
    H = ref.reshape(X.shape[0], X.shape[1], -1)
    W = ((R.permute(0, 2, 1) @ R + torch.eye(2)[None, :, :].cuda() * 1) * 1).inverse() @ (R.permute(0, 2, 1) @ H)
    W = F.relu(W.reshape(W.shape[0], W.shape[1], X.shape[2], X.shape[3]))
    F_x = -(W[:, 0:1] - W[:, 1:2])
    F_x = F_x.expand_as(X)

    return F_x, F_dx


def reg_scale(a, b):
    dim_range = list(range(1, a.dim()))
    return a * safe_divide(b.sum(dim=dim_range, keepdim=True), a.sum(dim=dim_range, keepdim=True))


def minmax_dims(x, minmax):
    y = x.clone()
    dims = x.dim()
    for i in range(1, dims):
        y = getattr(y, minmax)(dim=i, keepdim=True)[0]
    return y


def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, input, output):
    if hasattr(self, 'X'):
        del self.X

    if hasattr(self, 'Y'):
        del self.Y

    self.reshape_gfn = None

    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True

        if type(output) is torch.Tensor:
            if input[0].grad_fn is not None:
                input_input = input[0].grad_fn(self.X)
                if type(input_input) is torch.Tensor:
                    input_dims = input_input.dim()
                    output_dims = output.dim()
                    if input_dims != output_dims and input_dims == 4:
                        self.reshape_gfn = input[0].grad_fn

    self.Y = output


def delta_shift(C, R):
    dim_range = list(range(1, R.dim()))
    nonzero = C.ne(0).type(C.type())
    result = C + R - (R.sum(dim=dim_range, keepdim=True)) / (nonzero.sum(dim=dim_range, keepdim=True)) * nonzero

    return result


# Layers
class AGFProp(nn.Module):
    def __init__(self):
        super(AGFProp, self).__init__()
        self.register_forward_hook(forward_hook)

    def AGF(self, cam=None, grad_outputs=None, **kwargs):
        # Gradient
        Y = self.forward(self.X)
        S = grad_outputs
        grad_out = torch.autograd.grad(Y, self.X, S)

        return cam, grad_out


class AGFPropSimple(AGFProp):
    def AGF(self, cam=None, grad_outputs=None, **kwargs):
        def backward(cam):
            Z = self.forward(self.X) * cam.ne(0).type(cam.type())
            S = safe_divide(cam, Z)

            if torch.is_tensor(self.X) == False:
                result = []
                grad = torch.autograd.grad(Z, self.X, S)
                result.append(self.X[0] * grad[0])
                result.append(self.X[1] * grad[1])
            else:
                grad = torch.autograd.grad(Z, self.X, S)
                result = self.X * grad[0]
            return result

        if torch.is_tensor(cam) == False:
            idx = len(cam)
            tmp_cam = cam
            result = []
            for i in range(idx):
                cam_tmp = backward(tmp_cam[i])
                result.append(cam_tmp)
        else:
            result = backward(cam)

        # Gradient
        Y = self.forward(self.X)
        S = grad_outputs
        grad_out = torch.autograd.grad(Y, self.X, S)

        return result, grad_out


class Add(AGFPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)


class Clone(AGFProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def AGF(self, cam=None, grad_outputs=None, **kwargs):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)

        S = []

        for z, c in zip(Z, cam):
            S.append(safe_divide(c, z))

        C = torch.autograd.grad(Z, self.X, S)

        R = self.X * C[0]

        # Gradient
        Y = self.forward(self.X, self.num)
        S = grad_outputs
        grad_out = torch.autograd.grad(Y, self.X, S)

        return R, grad_out


class Cat(AGFProp):
    def forward(self, inputs, dim):
        self.__setattr__('dim', dim)
        return torch.cat(inputs, dim)

    def AGF(self, cam=None, grad_outputs=None, **kwargs):
        raise NotImplemented


class ReLU(nn.ReLU, AGFPropSimple):
    pass


class MaxPool2d(nn.MaxPool2d, AGFPropSimple):
    pass


class Dropout(nn.Dropout, AGFProp):
    pass


class BatchNorm2d(nn.BatchNorm2d, AGFProp):
    pass


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, AGFPropSimple):
    pass


class AvgPool2d(nn.AvgPool2d, AGFPropSimple):
    pass


class Linear(nn.Linear, AGFProp):
    def AGF(self, cam=None, grad_outputs=None, **kwargs):
        if grad_outputs is None:
            # Salient
            Y = self.forward(self.X)

            if 'K' in kwargs.keys():
                target_class = Y.data.topk(kwargs['K'], dim=1)[1]
            else:
                target_class = Y.data.topk(1, dim=1)[1]
            if 'index' in kwargs.keys():
                target_class = target_class[:, kwargs['index']:kwargs['index'] + 1]
            if 'class_id' in kwargs.keys():
                if type(kwargs['class_id']) is list:
                    assert len(kwargs['class_id']) == len(target_class)
                    for i in range(len(kwargs['class_id'])):
                        target_class[i, 0] = kwargs['class_id'][i]
                else:
                    raise Exception('Must be a list')

            # Initial propagation
            tgt = torch.zeros_like(Y)
            tgt = tgt.scatter(1, target_class, 1)
            yt = Y.gather(1, target_class)
            sigma = (Y - yt).abs().max(dim=1, keepdim=True)[0]
            Y = F.softmax(yt * torch.exp(-0.5 * ((Y - yt) / sigma) ** 2), dim=1)

            # Gradients stream
            grad_out = torch.autograd.grad(Y, self.X, tgt)

            result = self.X * grad_out[0]
        else:
            # Compute - C
            xabs = self.X.abs()
            wabs = self.weight.abs()
            Zabs = F.linear(xabs, wabs) * cam.ne(0).type(cam.type())

            S = safe_divide(cam, Zabs)
            grad = torch.autograd.grad(Zabs, xabs, S)
            C = xabs * grad[0]

            # Compute - M
            Y = self.forward(self.X)
            S = grad_outputs[0]

            grad = torch.autograd.grad(Y, self.X, S)
            if self.reshape_gfn is not None:
                x = self.reshape_gfn(self.X)
                g = self.reshape_gfn(grad[0])
                M = x * g
                M = M.mean(dim=1, keepdim=True).expand_as(x)
                M = M.reshape_as(self.X)
                M = F.relu(M) * C.ne(0).type(C.type())
                M = safe_divide(M, minmax_dims(M, 'max'))

                # Delta shift
                result = delta_shift(C, M)
            else:
                result = C

            # Gradients stream
            Y = self.forward(self.X)
            S = grad_outputs
            grad_out = torch.autograd.grad(Y, self.X, S)

        return result, grad_out


class Conv2d(nn.Conv2d, AGFProp):
    def gradprop2(self, DY, weight):
        Z = self.forward(self.X)

        output_padding = self.X.size()[2] - (
                (Z.size()[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0])

        return F.conv_transpose2d(DY, weight, stride=self.stride, padding=self.padding, output_padding=output_padding)

    def AGF(self, cam=None, grad_outputs=None, **kwargs):
        def final_backward(R_p, pw, nw, X1):
            X = X1
            L = X * 0 + \
                torch.min(torch.min(torch.min(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = X * 0 + \
                torch.max(torch.max(torch.max(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding)

            Sp = safe_divide(R_p, Za)

            Rp = X * self.gradprop2(Sp, self.weight) - L * self.gradprop2(Sp, pw) - H * self.gradprop2(Sp, nw)
            return Rp

        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)

        if self.X.shape[1] == 3:
            return final_backward(cam, pw, nw, self.X), None

        # Compute - M
        Y = self.forward(self.X)
        S = grad_outputs[0]
        grad = torch.autograd.grad(Y, self.X, S)

        M = F.relu((self.X * grad[0]).mean(dim=1, keepdim=True).expand_as(self.X))
        M = safe_divide(M, minmax_dims(M, 'max'))

        # Type Grad
        Y = self.forward(self.X) * cam.ne(0).type(cam.type())
        S = grad_outputs[0]
        grad = torch.autograd.grad(Y, self.X, S)

        gradcam = self.X * F.adaptive_avg_pool2d(grad[0], 1)
        gradcam = gradcam.mean(dim=1, keepdim=True).expand_as(self.X)

        # Compute - C
        xabs = self.X.abs()
        wabs = self.weight.abs()
        Zabs = F.conv2d(xabs, wabs, bias=None, stride=self.stride, padding=self.padding) * cam.ne(0).type(cam.type())

        S = safe_divide(cam, Zabs)
        grad = torch.autograd.grad(Zabs, xabs, S)
        C = xabs * grad[0]

        # Compute Factorization - F_x, F_dx
        Y = self.forward(self.X) * cam.ne(0).type(cam.type())
        S = grad_outputs[0]
        grad = torch.autograd.grad(Y, self.X, S)

        F_x, F_dx = get_factorization(self.X, grad[0], C.mean(dim=1, keepdim=True))

        F_x = F.relu(F_x)
        F_dx = F.relu(F_dx)

        # Compute - A
        wabs = self.weight.abs()
        xabs = torch.ones_like(self.X)
        xabs.requires_grad_()
        Zabs = F.conv2d(xabs, wabs, bias=None, stride=self.stride, padding=self.padding) * cam.ne(0).type(cam.type())
        S = safe_divide(cam, Zabs)
        grad = torch.autograd.grad(Zabs, xabs, S)
        A = xabs * grad[0]

        # Compute residual - R
        R = 0

        numer = 0
        if "no_fx" not in kwargs.keys() or not kwargs["no_fx"]:
            numer += F_x

        if "no_m" not in kwargs.keys() or not kwargs["no_m"]:
            numer += M

        if "gradcam" in kwargs.keys() and kwargs["gradcam"]:
            numer = F.relu(gradcam)

        if "no_reg" not in kwargs.keys() or not kwargs["no_reg"]:
            R += safe_divide(numer, 1 + torch.exp(-C))
        else:
            R += numer

        if "no_fdx" not in kwargs.keys() or not kwargs["no_fdx"]:
            R += F_dx

        if "no_a" not in kwargs.keys() or not kwargs["no_a"]:
            R += A

        R = R * C.ne(0).type(C.type())

        # Delta shift
        result = delta_shift(C, R)

        if "flat" in kwargs.keys() and kwargs["flat"]:
            cam_nonzero = cam.ne(0).type(cam.type())
            xabs = self.X.abs()
            wabs = self.weight.abs()
            Zabs = F.conv2d(xabs, wabs, bias=None, stride=self.stride, padding=self.padding) * cam_nonzero
            S = safe_divide(cam, Zabs)
            result = xabs * torch.autograd.grad(Zabs, xabs, S)[0]

        # Gradient
        Y = self.forward(self.X)
        S = grad_outputs
        grad_out = torch.autograd.grad(Y, self.X, S)

        return result, grad_out


class Sequential(nn.Sequential):
    def AGF(self, cam=None, grad_outputs=None, **kwargs):
        for m in reversed(self._modules.values()):
            cam, grad_outputs = m.AGF(cam, grad_outputs, **kwargs)
        return cam, grad_outputs
