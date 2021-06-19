import torch
from torch import nn
from tqdm import tqdm
from torch.nn.functional import interpolate
from augmentations import CenterCrop
import numpy as np
import os

def test(model, test_loader, splitter, epoch, evaluator, config, writer=None, dumper=None, final_mean=True):

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

                evaluator.compute_metrics(output, labels)

                # TB DUMP
                if writer is not None:
                    unempty_idx = np.argwhere(np.sum(labels != 2, axis=(0, 2)) > 0)
                    randidx = np.random.randint(0, unempty_idx.size - 1, 5)
                    rand_unempty_idx = unempty_idx[randidx].squeeze()  # random slices from unempty ones

                    dump_img = np.concatenate(np.moveaxis(images[:, rand_unempty_idx], 0, 1))
                    dump_img = dump_img * config['std'] + config['mean']

                    dump_gt = np.concatenate(np.moveaxis(labels[:, rand_unempty_idx], 0, 1))
                    dump_pred = np.concatenate(np.moveaxis(output[:, rand_unempty_idx], 0, 1))

                    dump_img = np.stack((dump_img, dump_img, dump_img), axis=-1)
                    a = dump_img.copy()
                    a[dump_pred == config['labels']['INSIDE']] = (1, 0, 0)
                    # a[dump_pred == config['labels']['CONTOUR']] = (0, 0, 1)
                    b = dump_img.copy()
                    b[dump_gt == config['labels']['INSIDE']] = (1, 0, 0)
                    # b[dump_gt == config['labels']['CONTOUR']] = (0, 0, 1)
                    dump_img = np.concatenate((a, b), axis=-2)
                    writer.add_image(
                        "2D_results",
                        dump_img,
                        epoch * len(test_loader) * config['batch_size'] / num_splits + patient_count,
                        dataformats='HWC'
                    )
                # END OF THE DUMP

                if dumper is not None:
                    dumper.dump(labels, output, images, name, score=evaluator.metric_list[-1])

                whole_image = whole_image[num_splits:]
                whole_output = whole_output[num_splits:]
                whole_names = whole_names[num_splits:]

    assert len(whole_output) == 0, "something wrong here"
    if final_mean:
        epoch_iou, epoch_dice = evaluator.mean_metric()
        return epoch_iou, epoch_dice
    else:
        return evaluator.iou_list


