import torchio as tio
import torch
from torch import nn
from tqdm import tqdm
from torch.nn.functional import interpolate
from augmentations import CropAndPad
import numpy as np
import os
import logging

def test2D(model, test_loader, epoch, writer, evaluator, type, splitter, config):

    num_splits = splitter.get_batch()
    whole_image, whole_output, whole_names = [], [], []
    patient_count = 0
    model.eval()
    with torch.no_grad():
        evaluator.reset_eval()
        for i, (images, _, sparse_gt, names) in tqdm(enumerate(test_loader), total=len(test_loader), desc='val epoch {}'.format(str(epoch))):

            images = images.cuda()  # BS, 3, H, W
            sparse_gt = sparse_gt.cuda()

            output = model(images, sparse_gt)  # BS, Classes, H, W

            if isinstance(output, tuple):
                output, _, _ = output

            whole_image += list(images.unsqueeze(-2).cpu())
            whole_output += list(output.unsqueeze(-2).cpu())
            whole_names += names

            while len(whole_image) >= num_splits:  # volume completed. let's evaluate it

                images = splitter.merge(whole_image[:num_splits])
                output = splitter.merge(whole_output[:num_splits])  # Classes, Z, H, W
                name = set(whole_names[:num_splits])  # keep unique names from this sub-volume
                assert len(name) == 1, "mixed patients!"  # must be just one!
                patient_count += 1
                name = name.pop()
                labels = np.load(os.path.join(config['file_path'], name))

                D, H, W = labels.shape[-3:]
                rD, rH, rW = output.shape[-3:]
                tmp_ratio = np.array((D / W, H / W, 1))
                pad_factor = tmp_ratio / np.array((rD / rW, rH / rW, 1))
                pad_factor /= np.max(pad_factor)
                reshape_size = np.array((D, H, W)) / pad_factor
                reshape_size = np.round(reshape_size).astype(np.int)

                output = interpolate(output.unsqueeze(0), size=tuple(reshape_size), mode='trilinear', align_corners=False).squeeze()  # (classes, Z, H, W) or (Z, H, W) if classes = 1
                output = CenterCrop((D, H, W))(output)

                images = interpolate(images.view(1, *images.shape), size=tuple(reshape_size), mode='trilinear', align_corners=False).squeeze()
                images = CenterCrop((D, H, W))(images)[0]  # Z, H W

                # final predictions
                if output.ndim > 3:
                    output = torch.argmax(torch.nn.Softmax(dim=0)(output), dim=0).numpy()
                else:
                    output = nn.Sigmoid()(output)  # Z, H, W
                    output[output > .5] = 1
                    output[output != 1] = 0
                    output = output.squeeze().cpu().detach().numpy()  # Z, H, W
                labels = labels.squeeze()
                images = images.numpy()

                evaluator.compute_metrics(output, labels, images, name, type)

                # TB DUMP
                # if writer is not None:
                #     unempty_idx = np.argwhere(np.sum(labels != 2, axis=(0, 2)) > 0)
                #     randidx = np.random.randint(0, unempty_idx.size - 1, 5)
                #     rand_unempty_idx = unempty_idx[randidx].squeeze()  # random slices from unempty ones
                #
                #     dump_img = np.concatenate(np.moveaxis(images[:, rand_unempty_idx], 0, 1))
                #     dump_img = dump_img * config['std'] + config['mean']
                #
                #     dump_gt = np.concatenate(np.moveaxis(labels[:, rand_unempty_idx], 0, 1))
                #     dump_pred = np.concatenate(np.moveaxis(output[:, rand_unempty_idx], 0, 1))
                #
                #     dump_img = np.stack((dump_img, dump_img, dump_img), axis=-1)
                #     a = dump_img.copy()
                #     a[dump_pred == config['labels']['INSIDE']] = (1, 0, 0)
                #     # a[dump_pred == config['labels']['CONTOUR']] = (0, 0, 1)
                #     b = dump_img.copy()
                #     b[dump_gt == config['labels']['INSIDE']] = (1, 0, 0)
                #     # b[dump_gt == config['labels']['CONTOUR']] = (0, 0, 1)
                #     dump_img = np.concatenate((a, b), axis=-2)
                #     writer.add_image(
                #         "2D_results",
                #         dump_img,
                #         epoch * len(test_loader) * config['batch_size'] / num_splits + patient_count,
                #         dataformats='HWC'
                #     )
                # END OF THE DUMP

                whole_image = whole_image[num_splits:]
                whole_output = whole_output[num_splits:]
                whole_names = whole_names[num_splits:]

    assert len(whole_output) == 0, "something wrong here"
    epoch_iou, epoch_dice, epoch_haus = evaluator.mean_metric(phase=type)
    if writer is not None and type != "Final":
        writer.add_scalar(f'{type}/IoU', epoch_iou, epoch)
        writer.add_scalar(f'{type}/Dice', epoch_dice, epoch)
        writer.add_scalar(f'{type}/Hauss', epoch_haus, epoch)

    logging.info(
        f'{type} Epoch [{epoch}], '
        f'{type} Mean Metric (IoU): {epoch_iou}'
        f'{type} Mean Metric (Dice): {epoch_dice}'
        f'{type} Mean Metric (haus): {epoch_haus}'
    )

    return epoch_iou, epoch_dice, epoch_haus


def test3D(model, test_loader, epoch, writer, evaluator, type):

    model.eval()

    with torch.no_grad():
        evaluator.reset_eval()
        for i, (subject, loader) in tqdm(enumerate(test_loader), total=len(test_loader), desc='val epoch {}'.format(str(epoch))):
            aggr = tio.inference.GridAggregator(subject, overlap_mode='average')
            for subvolume in loader:
                images = subvolume['data'][tio.DATA].float().cuda()  # BS, 3, Z, H, W
                sparse = subvolume['sparse'][tio.DATA].float().cuda()
                emb_codes = subvolume[tio.LOCATION].float().cuda()

                output = model(images, sparse, emb_codes)  # BS, Classes, Z, H, W

                aggr.add_batch(output, subvolume[tio.LOCATION])

            output = aggr.get_output_tensor()

            labels = np.load(subject[0]['gt_path'])  # original labels from storage
            images = np.load(subject[0]['data_path'])  # high resolution image from storage

            orig_shape = labels.shape[-3:]
            output = CropAndPad(orig_shape)(output).squeeze()  # keep pad_val = min(output) since we are dealing with probabilities

            # final predictions
            if output.ndim > 3:
                output = torch.argmax(torch.nn.Softmax(dim=0)(output), dim=0).numpy()
            else:
                output = nn.Sigmoid()(output)  # BS, 1, Z, H, W
                output[output > .5] = 1
                output[output != 1] = 0
                output = output.squeeze().cpu().detach().numpy()  # BS, Z, H, W

            evaluator.compute_metrics(output, labels, images, subject[0]['folder'], type)

    epoch_iou, epoch_dice, epoch_haus = evaluator.mean_metric(phase=type)
    if writer is not None and type != "Final":
        writer.add_scalar(f'{type}/IoU', epoch_iou, epoch)
        writer.add_scalar(f'{type}/Dice', epoch_dice, epoch)
        writer.add_scalar(f'{type}/Hauss', epoch_haus, epoch)

    if type in ['Test', 'Final']:
        logging.info(
            f'{type} Epoch [{epoch}], '
            f'{type} Mean Metric (IoU): {epoch_iou}'
            f'{type} Mean Metric (Dice): {epoch_dice}'
            f'{type} Mean Metric (haus): {epoch_haus}'
        )

    return epoch_iou, epoch_dice, epoch_haus


def inference(model, test_loader):

    model.eval()

    with torch.no_grad():
        for i, (subject, loader) in tqdm(enumerate(test_loader), total=len(test_loader)):
            aggr = tio.inference.GridAggregator(subject, overlap_mode='average')
            for subvolume in loader:
                images = subvolume['data'][tio.DATA].float().cuda()  # BS, 3, Z, H, W
                sparse = subvolume['sparse'][tio.DATA].float().cuda()
                emb_codes = subvolume[tio.LOCATION].float().cuda()

                output = model(images, sparse, emb_codes)  # BS, Classes, Z, H, W

                aggr.add_batch(output, subvolume[tio.LOCATION])

            output = aggr.get_output_tensor()


            orig_images = np.load(subject[0]['data_path'])  # high resolution image from storage

            orig_shape = orig_images.shape[-3:]
            output = CropAndPad(orig_shape)(output).squeeze()  # keep pad_val = min(output) since we are dealing with probabilities

            # final predictions
            if output.ndim > 3:
                output = torch.argmax(torch.nn.Softmax(dim=0)(output), dim=0).numpy()
            else:
                output = nn.Sigmoid()(output)  # BS, 1, Z, H, W
                output[output > .5] = 1
                output[output != 1] = 0
                output = output.squeeze().cpu().detach().numpy()  # BS, Z, H, W

            np.save(os.path.join("/nas/softechict-nas-2/mcipriano/datasets/maxillo/SPARSE", subject[0]['folder'], "generated.npy"), output)
            logging.info(f"patient {subject[0]['folder']} completed.")
