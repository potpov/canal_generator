import torch
import logging
from tqdm import tqdm
from torch import nn
import torchio as tio


def train2D(model, train_loader, loss_fn, optimizer, epoch, writer, evaluator, type='Train', warmup=1):

    model.train()
    evaluator.reset_eval()
    losses = []
    for i, (images, labels, sparse_gt, names) in tqdm(enumerate(train_loader), total=len(train_loader), desc='train epoch {}'.format(str(epoch))):

        images = images.cuda()
        labels = labels.cuda()
        sparse_gt = sparse_gt.cuda()

        optimizer.zero_grad()

        outputs = model(images, sparse_gt)  # BS, Classes, H, W

        loss = loss_fn(outputs, labels, warmup)

        if isinstance(outputs, tuple):
            outputs, std, mu = outputs

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # final predictions
        if outputs.shape[1] > 1:
            outputs = torch.argmax(torch.nn.Softmax(dim=1)(outputs), dim=1).cpu().numpy()
        else:
            outputs = nn.Sigmoid()(outputs)  # BS, 1, H, W
            outputs[outputs > .5] = 1
            outputs[outputs != 1] = 0
            outputs = outputs.squeeze().cpu().detach().numpy()  # BS, H, W

        labels = labels.squeeze().cpu().numpy()  # BS, Z, H, W
        evaluator.compute_metrics(outputs, labels, images, names, type)

    epoch_train_loss = sum(losses) / len(losses)
    epoch_iou, epoch_dice, epoch_haus = evaluator.mean_metric(phase=type)
    if writer is not None:
        writer.add_scalar(f'Loss/{type}', epoch_train_loss, epoch)
        writer.add_scalar(f'{type}', epoch_iou, epoch)

    logging.info(
        f'{type} Epoch [{epoch}], '
        f'{type} Mean Loss: {epoch_train_loss}, '
        f'{type} Mean Metric (IoU): {epoch_iou}'
        f'{type} Mean Metric (Dice): {epoch_dice}'
        f'{type} Mean Metric (haus): {epoch_haus}'
    )

    return epoch_train_loss, epoch_iou


def train3D(model, train_loader, loss_fn, optimizer, epoch, writer, evaluator, type='Train'):

    model.train()
    evaluator.reset_eval()
    losses = []
    for i, d in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'{type} epoch {str(epoch)}'):
        images = d['data'][tio.DATA].float().cuda()
        sparse = d['sparse'][tio.DATA].float().cuda()
        labels = d['label'][tio.DATA].cuda()
        emb_codes = torch.cat((
            d['index_ini'],
            d['index_ini'] + torch.as_tensor(images.shape[-3:])
        ), dim=1).float().cuda()
        optimizer.zero_grad()
        outputs = model(images, sparse, emb_codes)  # BS, Classes, Z, H, W
        loss = loss_fn(outputs, labels, 1)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        # final predictions
        if outputs.shape[1] > 1:
            outputs = torch.argmax(torch.nn.Softmax(dim=1)(outputs), dim=1).cpu().numpy()
        else:
            outputs = nn.Sigmoid()(outputs)  # BS, 1, Z, H, W
            outputs[outputs > .5] = 1
            outputs[outputs != 1] = 0
            outputs = outputs.squeeze().cpu().detach().numpy()  # BS, Z, H, W

        labels = labels.squeeze().cpu().numpy()  # BS, Z, H, W
        evaluator.compute_metrics(outputs, labels, images, d['folder'], type)

    epoch_train_loss = sum(losses) / len(losses)
    epoch_iou, epoch_dice, epoch_haus = evaluator.mean_metric(phase=type)
    if writer is not None:
        writer.add_scalar(f'Loss/{type}', epoch_train_loss, epoch)
        writer.add_scalar(f'{type}', epoch_iou, epoch)

    logging.info(
        f'{type} Epoch [{epoch}], '
        f'{type} Mean Loss: {epoch_train_loss}, '
        f'{type} Mean Metric (IoU): {epoch_iou}'
        f'{type} Mean Metric (Dice): {epoch_dice}'
        f'{type} Mean Metric (haus): {epoch_haus}'
    )

    return epoch_train_loss, epoch_iou