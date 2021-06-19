import argparse
import os
import pathlib
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

def save_weights(epoch, model, optim, score, path):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
        'metric': score,
    }
    torch.save(state, path)

def main(experiment_name):

    assert torch.cuda.is_available()
    logging.info(f"This model will run on {torch.cuda.get_device_name(torch.cuda.current_device())}")

    seed = config.get('seed', 47)
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader_config = config.get('data-loader', None)
    train_config = config.get('trainer', None)
    model_config = config.get('model')

    num_classes = 1 if len(loader_config['labels']) <= 2 else len(loader_config['labels'])
    model = utils.load_model(config)

    ngpus = torch.cuda.device_count()
    logging.info("going to use {} GPUs".format(ngpus))
    if model_config.get('sharding', False):
        logging.info('using model parallel')
        assert ngpus > 2
    else:
        logging.info('using data parallel')
        model = nn.DataParallel(model).cuda()

    train_params = model.parameters()

    optim_config = config.get('optimizer')
    optim_name = optim_config.get('name', None)
    if not optim_name or optim_name == 'Adam':
        optimizer = torch.optim.Adam(params=train_params, lr=optim_config['learning_rate'])
    elif optim_name == 'SGD':
        optimizer = torch.optim.SGD(params=train_params, lr=optim_config['learning_rate'])
    else:
        raise Exception("optimizer not recognized")

    sched_config = config.get('lr_scheduler')
    scheduler_name = sched_config.get('name', None)
    if scheduler_name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=sched_config['milestones'],
            gamma=sched_config['factor'],
        )
    elif scheduler_name == 'Plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=7)
    else:
        scheduler = None

    evaluator = Evaluator(loader_config)

    alveolar_data = AlveolarDataloader(config=loader_config)
    train_id, test_id, val_id = alveolar_data.split_dataset()

    train_loader = data.DataLoader(
        alveolar_data,
        batch_size=loader_config['batch_size'],
        sampler=SubsetRandomSampler(train_id),
        num_workers=loader_config['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    test_loader = data.DataLoader(
        alveolar_data,
        batch_size=loader_config['batch_size'],
        sampler=test_id,
        num_workers=loader_config['num_workers'],
        pin_memory=True,
        drop_last=False,
        collate_fn=alveolar_data.custom_collate
    )

    val_loader = data.DataLoader(
        alveolar_data,
        batch_size=loader_config['batch_size'],
        sampler=val_id,
        num_workers=loader_config['num_workers'],
        pin_memory=True,
        drop_last=False,
        collate_fn=alveolar_data.custom_collate
    )

    loss = LossFn(config.get('loss'), loader_config, weights=alveolar_data.get_weights())

    current_epoch = 0
    if train_config['checkpoint_path'] is not None:
        try:
            checkpoint = torch.load(train_config['checkpoint_path'])
            model.load_state_dict(checkpoint['state_dict'])
            current_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info(f"Checkpoint loaded successfully at epoch {current_epoch}, score:{checkpoint.get('metric', 'unavailable')})")
        except OSError as e:
            logging.info("No checkpoint exists from '{}'. Skipping...".format(train_config['checkpoint_path']))

    vol_writer = utils.SimpleDumper(loader_config, experiment_name, project_dir) if args.dump_results else None

    if train_config['do_train']:
        writer = SummaryWriter(log_dir=os.path.join(config['tb_dir'], experiment_name), purge_step=current_epoch)

        best_metric = 0
        warm_up = np.ones(shape=train_config['epochs'])
        wup_start_epoch = 5  # before this epoch warmup is 0, then it gradually grows
        warm_up[:wup_start_epoch] = 0
        warm_up[wup_start_epoch:wup_start_epoch+int(train_config['epochs'] * train_config.get('warm_up_length', 0.35))] = np.linspace(
            0, 1, num=int(train_config['epochs'] * train_config.get('warm_up_length', 0.35))
        )

        for epoch in range(current_epoch, train_config['epochs']):

            epoch_loss, _ = train(model, train_loader, loss, optimizer, epoch, writer, evaluator, warm_up[epoch])

            val_iou, val_dice = test(model, val_loader, alveolar_data.get_splitter(), epoch, evaluator, loader_config, writer=writer)
            logging.info(f'VALIDATION Epoch [{epoch}] - Mean Metric (iou): {val_iou} - (dice) {val_dice}')
            writer.add_scalar('Metric/validation', val_iou, epoch)

            if scheduler is not None:
                if optim_name == 'SGD' and scheduler_name == 'Plateau':
                    scheduler.step(val_iou)
                else:
                    scheduler.step(current_epoch)

            if val_iou > best_metric:
                best_metric = val_iou
                save_weights(epoch, model, optimizer, best_metric, os.path.join(project_dir, 'best.pth'))

            if val_iou < 1e-05 and epoch > 10:
                logging.info('drop in performances detected. aborting the experiment')
                return 0
            else:  # save current weights for debug, overwrite the same file
                save_weights(epoch, model, optimizer, val_iou, os.path.join(project_dir, 'checkpoints', 'last.pth'))

            if epoch % 5 == 0:
                test_iou, test_dice = test(model, test_loader, alveolar_data.get_splitter(), train_config['epochs'] + 1, evaluator, loader_config)
                logging.info(f'TEST Epoch [{epoch}] - Mean Metric (iou): {test_iou} - (dice) {test_dice}')
                writer.add_scalar('Metric/Test', test_iou, epoch)
                writer.add_scalar('Metric/Test_dice', test_dice, epoch)

        logging.info('BEST METRIC IS {}'.format(best_metric))
    else:
        writer = SummaryWriter(log_dir=os.path.join(config['tb_dir'], experiment_name))  # do not purge if no training was performed

    test_scores = test(
        model,
        test_loader,
        alveolar_data.get_splitter(),
        train_config['epochs'],
        evaluator,
        loader_config,
        writer=writer,
        dumper=vol_writer,
        final_mean=False
    )
    logging.info(f'final metric list: {test_scores}')
    if len(test_scores) > 0:
        logging.info(f'FINAL TEST - Mean Metric: {np.mean(test_scores)}')
    if vol_writer is not None:
        logging.info("going to create zip archive. wait the end of the run pls")
        vol_writer.save_zip()


if __name__ == '__main__':
    # execute the following line if there are new data in the dataset to be fixed
    # utils.fix_dataset_folder(r'Y:\work\datasets\maxillo\VOLUMES')

    RESULTS_DIR = r'Y:\work\results' if socket.gethostname() == 'DESKTOP-I67J6KK' else r'/nas/softechict-nas-2/mcipriano/results/maxillo/3D'
    BASE_YAML_PATH = os.path.join('configs', 'config.yaml') if socket.gethostname() == 'DESKTOP-I67J6KK' else os.path.join('configs', 'remote_config.yaml')

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--base_config', default="config.yaml", help='path to the yaml config file')
    arg_parser.add_argument('--verbose', action='store_true', help="if true sdout is not redirected, default: false")
    arg_parser.add_argument('--dump_results', action='store_true', help="dump test data, default: false")
    arg_parser.add_argument('--test', action='store_true', help="set up test params, default: false")

    args = arg_parser.parse_args()
    yaml_path = args.base_config

    if path.exists(yaml_path):
        print(f"loading config file in {yaml_path}")
        config = utils.load_config_yaml(yaml_path)
        experiment_name = config.get('title')
        project_dir = os.path.join(RESULTS_DIR, experiment_name)
    else:
        config = utils.load_config_yaml(BASE_YAML_PATH)  # load base config (remote or local)
        experiment_name = config.get('title', 'test')
        print('this experiment is on debug. no folders are going to be created.')
        project_dir = os.path.join(RESULTS_DIR, 'test')

    log_dir = pathlib.Path(os.path.join(project_dir, 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    if not args.verbose:
        # redirect streams to project dir
        sys.stdout = open(os.path.join(log_dir, 'std.log'), 'a+')
        sys.stderr = sys.stdout
        utils.set_logger(os.path.join(log_dir, 'logging.log'))
    else:
        # not create folder here, just log to console
        utils.set_logger()

    if args.test:
        config['trainer']['do_train'] = False
        config['data-loader']['num_workers'] = 0
        config['trainer']['checkpoint_path'] = os.path.join(project_dir, 'best.pth')

    main(experiment_name)