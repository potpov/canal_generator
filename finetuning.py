import argparse
import os
from tqdm import tqdm
import torch
import torch.utils.data as data
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import utils
from dataset import AlveolarDataloader
from eval import Eval as Evaluator
from losses import LossFn
from test import test
from train import train
import sys
import logging
import torch.nn as nn
import yaml
import numpy as np
from os import path
import socket
from skimage import measure
from matplotlib import pyplot as plt
import torchvision

RESULTS_DIR = r'Y:\work\results' if socket.gethostname() == 'DESKTOP-I67J6KK' else r'/nas/softechict-nas-2/mcipriano/results/maxillo/3D'

if __name__ == '__main__':
    # execute the following line if there are new data in the dataset to be fixed
    # utils.fix_dataset_folder(r'Y:\work\datasets\maxillo\VOLUMES')

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_name', default="config.yaml", help='path to the yaml config file')
    arg_parser.add_argument('--verbose', action='store_true', help="if true sdout is not redirected, default: false")

    args = arg_parser.parse_args()
    config = utils.load_config_yaml(os.path.join(RESULTS_DIR, args.exp_name, 'logs', 'config.yaml'))

    if not args.verbose:
        # redirect streams to project dir
        sys.stdout = open(os.path.join(RESULTS_DIR, args.exp_name, 'logs', 'std.log'), 'a+')
        sys.stderr = sys.stdout
        utils.set_logger(os.path.join(RESULTS_DIR, args.exp_name, 'logs', 'tuning.log'))
    else:
        # not create folder here, just log to console
        utils.set_logger()

    assert torch.cuda.is_available()
    logging.info("FINETUNING WITH TOPOLOGICAL LOSS (+ MSE)")
    logging.info(f"This model will run on {torch.cuda.get_device_name(torch.cuda.current_device())}")

    seed = config.get('seed', 47)
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader_config = config.get('data-loader', None)
    train_config = config.get('trainer', None)
    model_config = config.get('model')

    num_classes = 1 if len(loader_config['labels']) <= 2 else len(loader_config['labels'])

    model = utils.load_model(model_config,num_classes=num_classes,)

    model = nn.DataParallel(model).cuda()

    train_params = model.parameters()
    optimizer = torch.optim.SGD(params=train_params, lr=0.0001)
    # optimizer = torch.optim.Adam(params=train_params, lr=0.001)

    evaluator = Evaluator(loader_config)

    alveolar_data = AlveolarDataloader(config=loader_config, tuning=True)
    train_id, test_id, _ = alveolar_data.split_dataset()

    val_loader = data.DataLoader(
        alveolar_data,
        batch_size=loader_config['batch_size'],
        sampler=train_id,
        num_workers=loader_config['num_workers'],
        pin_memory=True,
        drop_last=False,
    )

    loss_fn = LossFn({'name': ['TopoLoss', 'MSE']}, loader_config, weights=alveolar_data.get_weights())

    checkpoint = torch.load(train_config['checkpoint_path'])
    model.load_state_dict(checkpoint['state_dict'])

    # test_scores = test(model, test_loader, alveolar_data.get_splitter(), 1, evaluator, loader_config, dumper=None, final_mean=False)
    # logging.info(f'before finetuning metric list: {test_scores}')
    # logging.info(f'before finetuning - Mean Metric: {np.mean(test_scores)}')

    writer = SummaryWriter(log_dir=os.path.join(config['tb_dir'], args.exp_name), purge_step=0)

    ##########
    # TRAINING

    model.train()

    for epoch in range(10):
        evaluator.reset_eval()
        a_losses = []
        b_losses = []

        for i, (images, labels, _) in tqdm(enumerate(val_loader), total=len(val_loader)):

            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = model(images)  # BS, Classes, H, W

            topo_loss = loss_fn.factory_loss(outputs, labels, 'TopoLoss', 1)
            jac_loss = loss_fn.factory_loss(outputs, labels, 'Jaccard', 1)

            loss = jac_loss + topo_loss * 0.01
            loss.backward()
            optimizer.step()

            a_losses.append(topo_loss.item())
            b_losses.append(jac_loss.item())

            # final predictions
            outputs = nn.Sigmoid()(outputs)  # BS, 1, H, W
            outputs[outputs > .5] = 1
            outputs[outputs != 1] = 0

            if i == 2:
                writer.add_image(f'topotest', torchvision.utils.make_grid(outputs, pad_value=1), epoch)

            outputs = outputs.squeeze().cpu().detach().numpy()  # BS, H, W
            evaluator.iou(outputs, labels.squeeze().cpu().numpy())

        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'metric': evaluator.metric_list[-1]}
        torch.save(state, os.path.join(RESULTS_DIR, args.exp_name, 'tuned.pth'))

        writer.add_scalar('Tuning/Score', evaluator.mean_metric(), epoch)
        writer.add_scalar('Tuning/LossJaccard', np.mean(b_losses), epoch)
        writer.add_scalar('Tuning/LossTopo', np.mean(a_losses), epoch)

    #############
    # TEST THE RESULTS

    test_loader = data.DataLoader(
        alveolar_data,
        batch_size=loader_config['batch_size'],
        sampler=test_id,
        num_workers=loader_config['num_workers'],
        pin_memory=True,
        drop_last=False,
        collate_fn=alveolar_data.custom_collate
    )

    test_scores = test(model, test_loader, alveolar_data.get_splitter(), 1, evaluator, loader_config, dumper=None, final_mean=False)
    logging.info(f'final metric list: {test_scores}')
    logging.info(f'FINAL TEST - Mean Metric: {np.mean(test_scores)}')





