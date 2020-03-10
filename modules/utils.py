import torch

__all__ = ['clrp_others', 'clrp_target', 'sglrp_others', 'sglrp_target', 'normalize']


# CLRP
def clrp_target(output, vis_class, **kwargs):
    if vis_class == 'top':
        pred = output.data.max(1, keepdim=True)[1]
        mask = torch.zeros_like(output)
        mask.scatter_(1, pred, 1)
    elif vis_class == 'index':
        mask = torch.zeros_like(output)
        mask[:, kwargs['class_id']] = 1
    elif vis_class == 'target':
        mask = torch.zeros_like(output)
        mask.scatter_(1, kwargs['target'], 1)
    else:
        raise Exception('Invalid vis-class')

    return mask * output


def clrp_others(output, vis_class, **kwargs):
    if vis_class == 'top':
        pred = output.data.max(1, keepdim=True)[1]
        mask = torch.ones_like(output)
        mask.scatter_(1, pred, 0)
        pred_out = output.gather(1, pred)
    elif vis_class == 'index':
        mask = torch.ones_like(output)
        mask[:, kwargs['class_id']] = 0
        pred_out = output[:, kwargs['class_id']:kwargs['class_id'] + 1]
    elif vis_class == 'target':
        mask = torch.ones_like(output)
        mask.scatter_(1, kwargs['target'], 0)
        pred_out = output.gather(1, kwargs['target'])
    else:
        raise Exception('Invalid vis-class')

    mask /= (output.shape[-1] - 1)

    return mask * pred_out


# SGLRP
def sglrp_target(output, vis_class, **kwargs):
    sm_pred = torch.softmax(output, dim=1)

    if vis_class == 'top':
        pred = output.data.max(1, keepdim=True)[1]
        mask = torch.zeros_like(output)
        mask.scatter_(1, pred, 1)
    elif vis_class == 'index':
        mask = torch.zeros_like(output)
        mask[:, kwargs['class_id']] = 1
    elif vis_class == 'target':
        mask = torch.zeros_like(output)
        mask.scatter_(1, kwargs['target'], 1)
    else:
        raise Exception('Invalid vis-class')

    return mask * (sm_pred * (1 - sm_pred) + 1e-8)


def sglrp_others(output, vis_class, **kwargs):
    sm_pred = torch.softmax(output, dim=1)

    if vis_class == 'top':
        pred = output.data.max(1, keepdim=True)[1]
        mask = torch.ones_like(output)
        mask.scatter_(1, pred, 0)
        sm_pred_out = sm_pred.gather(1, pred)
    elif vis_class == 'index':
        mask = torch.ones_like(output)
        mask[:, kwargs['class_id']] = 0
        sm_pred_out = sm_pred[:, kwargs['class_id']]
    elif vis_class == 'target':
        mask = torch.ones_like(output)
        mask.scatter_(1, kwargs['target'], 0)
        sm_pred_out = sm_pred.gather(1, kwargs['target'])
    else:
        raise Exception('Invalid vis-class')

    return mask * sm_pred_out * sm_pred


def normalize(tensor,
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor

