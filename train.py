import torch
import logging
from tqdm import tqdm
from torch import nn


def train(model, train_loader, loss_fn, optimizer, epoch, writer, evaluator, warmup, tuning=False):

    model.train()
    evaluator.reset_eval()
    losses = []
    for i, (images, labels, sparse_gt, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc='train epoch {}'.format(str(epoch))):

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
        evaluator.compute_metrics(outputs, labels)

    epoch_train_loss = sum(losses) / len(losses)
    epoch_iou, epoch_dice = evaluator.mean_metric()
    if writer is not None:
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Metric/train', epoch_iou, epoch)

    logging.info(
        f'Train Epoch [{epoch}], '
        f'Train Mean Loss: {epoch_train_loss}, '
        f'Train Mean Metric: {epoch_iou}'
        f'Train Mean Metric (Dice): {epoch_dice}'
    )

    return epoch_train_loss, epoch_iou