import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from topologylayer.nn import LevelSetLayer2D
from topologylayer.nn.features import get_raw_barcode_lengths


class TopLoss2D(nn.Module):
    def __init__(self, size, expected=1):
        super(TopLoss2D, self).__init__()
        self.size = size
        self.expected = expected
        self.pdfn = LevelSetLayer2D(self.size, maxdim=1, sublevel=False, complex='grid')

    def forward(self, data):
        assert data.shape[-2:] == self.size, 'check the shape!'
        loss = 0
        for img in data:

            dgminfo = self.pdfn(img)
            holes = dgminfo[0][1]
            holes[holes[:, 1] == -np.inf, 1] = 0  # fix the longest components bug (is it a real bug?)
            lenghts = get_raw_barcode_lengths(holes, False)  # get lens
            lenghts[lenghts != lenghts] = 0
            sortl, _ = torch.sort(lenghts, descending=True)  # from the biggest to the shortes

            L0 = (1. - sortl[:self.expected] ** 2).sum()
            L01 = torch.sum(sortl[self.expected:10] ** 2)
            loss = loss + L0 + L01
        return loss


def jaccard(outputs, targets, per_image=False, non_empty=False, min_pixels=5):
    batch_size = outputs.size()[0]
    eps = 1e-3
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    target_sum = torch.sum(dice_target, dim=1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    losses = 1 - (intersection + eps) / (torch.sum(dice_output + dice_target, dim=1) - intersection + eps)
    if non_empty:
        assert per_image == True
        non_empty_images = 0
        sum_loss = 0
        for i in range(batch_size):
            if target_sum[i] > min_pixels:
                sum_loss += losses[i]
                non_empty_images += 1
        if non_empty_images == 0:
            return 0
        else:
            return sum_loss / non_empty_images
    return losses.mean()


def kl_loss(mu, logvar):
    """
    compute the Kullback Leibler distance between the predicted distribution
    and the normal N(0, I)
    :param mu: predicted mean
    :param logvar: predicted log(variance)
    :return: kl_distance
    """
    # kl_element = torch.add(torch.add(torch.add(mu.pow(2), logvar.exp()), -1), logvar.mul(-1))
    kl_element = 1+logvar-mu.pow(2)-logvar.exp()
    kl_loss = torch.mean(torch.sum(kl_element * -0.5, axis=1))
    # ste kl
    # kl = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    return kl_loss

class JaccardLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False, non_empty=False, apply_sigmoid=False,
                 min_pixels=5):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image
        self.non_empty = non_empty
        self.apply_sigmoid = apply_sigmoid
        self.min_pixels = min_pixels

    def forward(self, input, target):
        if self.apply_sigmoid:
            input = torch.sigmoid(input)
        return jaccard(input, target, per_image=self.per_image, non_empty=self.non_empty, min_pixels=self.min_pixels)

def DiceLossv2(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


class DiceLoss(nn.Module):
    def __init__(self, classes, device):
        super().__init__()
        self.eps = 1e-06
        self.classes = classes
        self.device = device

    def forward(self, pred, gt):
        included = [v for k, v in self.classes.items() if k not in ['BACKGROUND', 'UNLABELED']]

        gt_onehot = one_hot_encode(gt, pred.shape, self.device)  # B, 1, (other dimensions) -> B, (other dimensions), C
        gt_onehot = gt_onehot.moveaxis(-1, 1)  # bring C to the correct position after batch size  -> B, C (other dimensions)
        input_soft = F.softmax(pred, dim=1)
        dims = tuple(np.arange(2, gt_onehot.ndim).astype(int))
        intersection = torch.sum(input_soft * gt_onehot, dims)
        cardinality = torch.sum(input_soft + gt_onehot, dims)
        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score[:, included])


# def one_hot_encode(volume, shape, device):
#     B, C, Z, H, W = shape
#     flat = volume.reshape(-1).unsqueeze(dim=1)  # 1xB*Z*H*W
#     onehot = torch.zeros(size=(B * Z * H * W, C), dtype=torch.float).to(device)  # 1xB*Z*H*W destination tensor
#     onehot.scatter_(1, flat, 1)  # writing the conversion in the destination tensor
#     return torch.squeeze(onehot).reshape(B, Z, H, W, C)  # reshaping to the original shape

def one_hot_encode(volume, shape, device):
    B, C = shape[:2]
    axes = torch.as_tensor(shape)[-2:]  # support 2D, 3D, nD
    tot_axes = torch.prod(axes)  # number of voxels excluding Batch and Channels/classes

    flat = volume.reshape(-1).unsqueeze(dim=1)  # 1xB*Z*H*W
    onehot = torch.zeros(size=(B * tot_axes, C), dtype=torch.float).to(device)  # 1xB*Z*H*W destination tensor
    onehot.scatter_(1, flat, 1)  # writing the conversion in the destination tensor
    return torch.squeeze(onehot).reshape(B, *list(axes), C)  # reshaping to the original shape


class LossFn:
    def __init__(self, loss_config, loader_config, weights):

        if not isinstance(loss_config['name'], list):
            self.name = [loss_config['name']]
        else:
            self.name = loss_config['name']
        self.loader_config = loader_config
        self.classes = loader_config['labels']
        self.weights = weights

    def factory_loss(self, pred, gt, name, warmup):

        if name == 'CrossEntropyLoss':
            # sigmoid here which is included in other losses
            pred = torch.nn.Sigmoid()(pred)
            loss_fn = nn.CrossEntropyLoss(weight=self.weights).to(self.device)
        elif name == 'BCEWithLogitsLoss':

            # VECCHIO ONE HOT
            # one hot encoding for cross entropy with digits. Bx1xHxW -> BxCxHxW
            # B, C, Z, H, W = pred.shape
            # gt_flat = gt.reshape(-1).unsqueeze(dim=1)  # 1xB*Z*H*W
            # gt_onehot = torch.zeros(size=(B * Z * H * W, C), dtype=torch.float).to(self.device)  # 1xB*Z*H*W destination tensor
            # gt_onehot.scatter_(1, gt_flat, 1)  # writing the conversion in the destination tensor
            # gt = torch.squeeze(gt_onehot).reshape(B, Z, H, W, C)  # reshaping to the original shape
            # pred = pred.permute(0, 2, 3, 4, 1)  # for BCE we want classes in the last axis

            # NUOVO ONE HOT
            gt = one_hot_encode(gt, pred.shape, self.device)
            pred = pred.moveaxis(1, -1)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.weights).to(self.device)
        elif name == 'Jaccard':
            assert pred.shape[1] == 1, 'this loss works with a binary prediction'
            return JaccardLoss(weight=self.weights, apply_sigmoid=True)(pred, gt)
        elif name == 'DiceLoss':
            # pred = torch.argmax(torch.nn.Softmax(dim=1)(pred), dim=1)
            # pred = pred.data.cpu().numpy()
            # gt = gt.cpu().numpy()
            return DiceLoss(self.classes, self.device)(pred, gt)
        elif name == 'DiceLossv2':
            return DiceLossv2(gt, pred)
        elif name == 'TopoLoss':
            # pred = nn.Sigmoid()(pred)  # i want probabilities - this makes the method unsuitable for multiclass!
            pred = F.softmax(pred, dim=1)

            contour_idxs = np.argwhere(np.sum(gt.cpu().numpy() == self.classes['CONTOUR'], axis=tuple(np.arange(1, gt.ndim))) > 0).squeeze()

            contour_loss = TopLoss2D(gt.shape[-2:], expected=1)
            topoloss = contour_loss(pred[contour_idxs[:4], self.classes['CONTOUR']].squeeze())

            return 0.1 * topoloss

        elif name == 'MSE':
            loss_fn = torch.nn.MSELoss()
        else:
            raise Exception("specified loss function cant be found.")

        return loss_fn(pred, gt)

    def __call__(self, outputs, gt, warmup):
        """
        SHAPE MUST BE Bx1xHxW
        :param pred:
        :param gt:
        :return:
        """
        cur_loss = []

        if isinstance(outputs, tuple):
            pred, std, mu = outputs
            cur_loss.append(kl_loss(mu, std) * warmup)
        else:
            pred = outputs

        assert pred.device == gt.device
        assert gt.device != 'cpu'
        self.device = pred.device

        for name in self.name:
            loss = self.factory_loss(pred, gt, name, warmup)
            if loss == 0:
                continue
            if torch.isnan(loss):
                raise ValueError('Loss is nan during training...')
            cur_loss.append(loss)
        return torch.sum(torch.stack(cur_loss))
