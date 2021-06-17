from torch.utils.data.dataset import Dataset
import numpy as np
from torchvision import transforms
import os
from matplotlib import pyplot as plt
from augmentations import RandomRotate, RandomContrast, ElasticDeformation, Normalize, ToTensor, CenterPad, RandomVerticalFlip, Resize, Rescale
import torch
import json
from tqdm import tqdm
from Jaw import Jaw
import logging
from utils import Splitter


class AlveolarDataloader(Dataset):

    def __init__(self, config, tuning=False):
        """
        data loader
        :param config: yaml file
        :param tuning: use validation as a training set
        """

        self.config = config

        self.patients = {
            'data': [],
            'gt': [],
            'sparse_gt': [],
            'name': []
        }
        self.indices = {
            'test': [],
            'train': [],
            'val': []
        }

        self.splitter = Splitter(config.get('split_volumes', (1, 128, 2)))

        self.dicom_max = config.get('volumes_max', 2100)
        self.dicom_min = config.get('volumes_min', 0)

        self.augmentation = transforms.Compose([
            RandomRotate(execution_probability=0.5, order=4),
            RandomVerticalFlip(execution_probability=0.7),
            # RandomContrast(execution_probability=0.5),
            # ElasticDeformation(execution_probability=0.2),
        ])

        reshape_size = self.config.get('resize_shape', (152, 224, 256))
        self.reshape_size = tuple(reshape_size) if type(reshape_size) == list else reshape_size

        gt_filename = 'gt_alpha_multi.npy' if len(self.config['labels']) > 2 else 'gt_alpha.npy'
        index = 0

        self.mean = self.config.get('mean', None)
        self.std = self.config.get('std', None)
        self.means = []
        self.stds = []

        with open(config.get('split_filepath')) as f:
            folder_splits = json.load(f)

        for partition, folders in folder_splits.items():

            for patient_num, folder in tqdm(enumerate(folders), total=len(folders)):
                data = np.load(os.path.join(config['file_path'], folder, 'data.npy'))
                sparse_gt = np.load(os.path.join(config['sparse_path'], folder, 'gt_sparse.npy'))
                gt = np.load(os.path.join(config['file_path'], folder, gt_filename))

                assert data.max() > 1  # data should not be normalized by default

                data, gt, sparse_gt = self.preprocessing(data, gt, sparse_gt, folder, partition=partition)
                self.patients['data'] += data
                self.patients['gt'] += gt
                self.patients['sparse_gt'] += sparse_gt
                self.patients['name'] += [os.path.join(folder, gt_filename) for i in data]  # replicating N times the name of the folder

                self.indices[partition] += list(range(index, index + len(data)))
                index = index + len(data)

        if self.mean is None or self.std is None:
            self.mean = np.mean(self.means)
            self.std = np.mean(self.stds)

        logging.info(f'mean for the dataset: {self.mean}, std: {self.std}')

        self.indices['train'] = np.asarray(self.indices['train'])
        self.indices['test'] = np.asarray(self.indices['test'])
        self.indices['val'] = np.asarray(self.indices['val'])

        self.weights = self.config.get('weights', None)
        if self.weights is None:
            logging.info('going to compute weights')
            self.weights = self.median_frequency_balancing()
        else:
            self.weights = torch.Tensor(self.weights)
        logging.info(f'weights for this dataset: {self.weights}')

        logging.info('folders in validation set: {}'.format(folder_splits.get('val', 'None')))
        logging.info('folders in test set: {}'.format(folder_splits['test']))

    def __len__(self):
        return self.indices['train'].size + self.indices['test'].size + self.indices['val'].size

    def get_weights(self):
        return self.weights

    def get_config(self):
        return self.config

    def preprocessing(self, data, gt, sparse_gt, folder, partition='train'):

        # rescale
        data = np.clip(data, self.dicom_min, self.dicom_max)
        data = (data.astype(np.float) + self.dicom_min) / (self.dicom_max + self.dicom_min)   # [0-1] with shifting

        if self.mean is None or self.std is None:
            self.means.append(np.mean(data))
            self.stds.append(np.std(data))

        D, H, W = data.shape[-3:]
        rD, rH, rW = self.reshape_size
        tmp_ratio = np.array((D/W, H/W, 1))
        pad_factor = tmp_ratio / np.array((rD/rW, rH/rW, 1))
        pad_factor /= np.max(pad_factor)
        new_shape = np.array((D, H, W)) / pad_factor
        new_shape = np.round(new_shape).astype(np.int)

        data = CenterPad(new_shape)(data)

        # suppress areas out of the splines
        # if self.config.get('background_suppression', True):
        #     data = utils.background_suppression(data, folder)
        #
        #     # cut the overflowing null areas -> extract cube with extreme limits of where are the values != 0
        #     xs, ys, zs = np.where(data != 0)
        #     data = data[min(xs):max(xs) + 1, min(ys):max(ys) + 1, min(zs):max(zs) + 1]
        #     if partition == 'train':
        #         gt = gt[min(xs):max(xs) + 1, min(ys):max(ys) + 1, min(zs):max(zs) + 1]

        data = Rescale(size=self.reshape_size)(data)

        sparse_gt = CenterPad(new_shape)(sparse_gt, pad_val=self.config['labels']['BACKGROUND'])
        sparse_gt = Rescale(size=self.reshape_size, interp_fn='nearest')(sparse_gt)

        if partition == 'train':
            gt = CenterPad(new_shape)(gt, pad_val=self.config['labels']['BACKGROUND'])
            gt = Rescale(size=self.reshape_size, interp_fn='nearest')(gt)
        else:
            gt = np.zeros_like(data)  # this is because in test and train we load gt at runtime

        data = self.splitter.split(data)
        sparse_gt = self.splitter.split(sparse_gt)
        gt = self.splitter.split(gt)

        return data, gt, sparse_gt

    def __getitem__(self, index):
        vol, gt = self.patients['data'][index].astype(np.float32), self.patients['gt'][index].astype(np.int64)
        sparse_gt = self.patients['sparse_gt'][index].astype(np.float32)

        # if index in self.indices['train']:
        #     vol, gt = self.augmentation([vol, gt])
        #     assert np.array_equal(gt, gt.astype(bool)), 'something wrong with augmentations here'
        vol = transforms.Normalize(self.mean, self.std)(ToTensor()(vol.copy()))
        gt = ToTensor()(gt.copy())
        sparse_gt = ToTensor()(sparse_gt.copy())
        vol = vol.repeat(3, 1, 1)  # creating the channel axis and making it RGB
        return vol, gt, sparse_gt, self.patients['name'][index]

    def get_splitter(self):
        return self.splitter

    def split_dataset(self):

        np.random.shuffle(self.indices['train'])
        return self.indices['train'], self.indices['test'], self.indices['val']

    def class_freq(self):
        """
        Computes class frequencies for each label.
        Returns the number of pixels of class c (in all images) divided by the total number of pixels (in images where c is present).
        Returns:
            (torch.Tensor): tensor with shape n_labels, with class frequencies for each label.
        """
        num_labels = len(self.config['labels'])
        class_pixel_count = torch.zeros(num_labels)
        total_pixel_count = torch.zeros(num_labels)

        for gt in self.patients['gt']:
            gt_ = torch.from_numpy(gt)
            counts = torch.bincount(gt_.flatten())
            class_pixel_count += counts
            n_pixels = gt_.numel()
            total_pixel_count = torch.where(counts > 0, total_pixel_count + n_pixels, total_pixel_count)

        return class_pixel_count / total_pixel_count

    def class_freq_2(self, valid_labels):

        num_labels = len(self.config.get('labels'))
        class_pixel_count = num_labels * [0]

        for idx in self.indices['train']:
            gt = self.patients['gt'][idx]
            for l in valid_labels:
                class_pixel_count[l] += np.sum(gt == l) / np.sum(np.in1d(gt, valid_labels))

        return [c / len(self.patients['gt']) for c in class_pixel_count]

    def median_frequency_balancing(self):
        """
        Computes class weights using Median Frequency Balancing.
        Source paper: https://arxiv.org/pdf/1411.4734.pdf (par. 6.3.2)
        Returns:
            (torch.Tensor): class weights
        """
        excluded = ['UNLABELED']
        valid_labels = [v for k, v in self.config.get('labels').items() if k not in excluded]
        freq = self.class_freq_2(valid_labels)
        # freq = self.class_freq()
        # sorted, _ = torch.sort(freq)
        median = torch.median(torch.Tensor([f for (i, f) in enumerate(freq) if i in valid_labels]))
        # median = torch.median(freq)
        weights = torch.Tensor([median / f if f != 0 else 0 for f in freq])
        weights /= weights.sum()  # normalizing
        return weights

    def pytorch_weight_sys(self):
        excluded = ['UNLABELED']
        valid_labels = [v for k, v in self.config.get('labels').items() if k not in excluded]

        class_pixel_count = torch.zeros(len(self.config.get('labels')))
        not_class_pixel_count = torch.zeros(len(self.config.get('labels')))
        for gt in self.patients['gt']:
            for l in valid_labels:
                class_pixel_count[l] += np.sum(gt == l)
                not_class_pixel_count[l] += np.sum(np.in1d(gt, [v for v in valid_labels if v != l]))

        return not_class_pixel_count / (class_pixel_count + 1e-06)

    def custom_collate(self, batch):
        images = torch.stack([item[0] for item in batch])
        labels = [item[1] for item in batch]
        sparse = torch.stack([item[2] for item in batch])
        names = [item[3] for item in batch]
        return images, labels, sparse, names
