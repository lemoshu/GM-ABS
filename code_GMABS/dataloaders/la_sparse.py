import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import torch.nn.functional as F
import torchio as tio


class LA_heart_sparse_SAM(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None, with_idx=False):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.with_idx = with_idx

        train_path = self._base_dir+'/train.txt'
        test_path = self._base_dir+'/test.txt'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

        # Initialize pseudo labels with None
        self.pseudo_labels = [None] * len(self.image_list)

    def __len__(self):
        return len(self.image_list)

    def update_pseudo_labels(self, name_list, new_pseudo_labels):
        """ Update the pseudo labels for the entire dataset. """
        assert len(new_pseudo_labels) == len(self.image_list)
        for image_name_ in self.image_list:
            idx_new = name_list.index(image_name_)
            self.pseudo_labels[self.image_list.index(image_name_)] = new_pseudo_labels[idx_new]     
        # check if each sample in pseudo_labels has consistent shape with its label
        # for idx in range(len(self.image_list)):
        #     if self.pseudo_labels[idx] is not None:
        #         print("pseudo label shape: ", self.pseudo_labels[idx].shape
        #               , "label shape: ", self.__getitem__(idx)['label'].shape) # pseudo label: (183, 140, 88)
        #         assert self.pseudo_labels[idx].shape == self.__getitem__(idx)['label'].shape # label: (112, 112, 80)
        
    # def __getitem__(self, idx):
    #     image_name = self.image_list[idx]
    #     h5f = h5py.File(self._base_dir + "/data/{}.h5".format(image_name), 'r')
    #     image = h5f['image'][:]
    #     label = h5f['label'][:]
    #     cross_label = h5f['cross_label'][:]
    #     indicator = h5f['labeled_slice_indicator'][:]
    #     sample = {'image': image, 'label': label.astype(np.uint8), 'cross_label': cross_label.astype(np.uint8),
    #               'indicator': indicator.astype(np.uint8)}
    #     pseudo_label = self.pseudo_labels[idx] if self.pseudo_labels[idx] is not None else np.zeros_like(sample['label'])
    #     sample['pseudo_label'] = pseudo_label.astype(np.uint8)
    #     if self.transform:
    #         sample = self.transform(sample)
    #     if self.with_idx:
    #         sample['idx'] = idx
    #     return sample

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/data/{}.h5".format(image_name), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        cross_label = h5f['cross_label'][:]
        indicator = h5f['labeled_slice_indicator'][:]
        sample = {'image': image, 'label': label.astype(np.uint8), 'cross_label': cross_label.astype(np.uint8),
                  'indicator': indicator.astype(np.uint8)}
        # add pseudo label into the sample dict
        pseudo_label = self.pseudo_labels[idx] if self.pseudo_labels[idx] is not None else np.zeros_like(sample['label'])
        sample = {'image': image, 'label': label.astype(np.uint8), 'cross_label': cross_label.astype(np.uint8),
                    'indicator': indicator.astype(np.uint8), 'pseudo_label': pseudo_label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        if self.with_idx:
            sample['idx'] = idx
        return sample


class LA_heart_sparse(Dataset):
    """ BraTS2019 Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None, with_idx=False):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.with_idx = with_idx

        train_path = self._base_dir+'/train.txt'
        test_path = self._base_dir+'/test.txt'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/data/{}.h5".format(image_name), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        cross_label = h5f['cross_label'][:]
        indicator = h5f['labeled_slice_indicator'][:]
        sample = {'image': image, 'label': label.astype(np.uint8), 'cross_label': cross_label.astype(np.uint8),
                  'indicator': indicator.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        if self.with_idx:
            sample['idx'] = idx
        return sample


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, cross_label, indicator = sample['image'], sample['label'], sample['cross_label'], sample['indicator']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            cross_label = np.pad(cross_label, [(pw, pw), (ph, ph), (pd, pd)],
                                    mode='constant', constant_values=0)
            indicator = np.pad(indicator, [(pw, pw), (ph, ph), (pd, pd)],
                                    mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        cross_label = cross_label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        indicator = indicator[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label, 'cross_label': cross_label, 'indicator': indicator}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label, cross_label, indicator = sample['image'], sample['label'], sample['cross_label'], sample['indicator']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            cross_label = np.pad(cross_label, [(pw, pw), (ph, ph), (pd, pd)],
                                    mode='constant', constant_values=0)
            indicator = np.pad(indicator, [(pw, pw), (ph, ph), (pd, pd)],
                                    mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)],
                             mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        cross_label = cross_label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        indicator = indicator[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label, 'cross_label': cross_label, 'indicator': indicator}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label, cross_label, indicator = sample['image'], sample['label'], sample['cross_label'], sample['indicator']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        cross_label = np.rot90(cross_label, k)
        indicator = np.rot90(indicator, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        cross_label = np.flip(cross_label, axis=axis).copy()
        indicator = np.flip(indicator, axis=axis).copy()
        return {'image': image, 'label': label, 'cross_label': cross_label, 'indicator': indicator}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(
            image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label, 'cross_label': sample['cross_label'], 'indicator': sample['indicator']}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros(
            (self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(
            1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(), 'cross_label': torch.from_numpy(sample['cross_label']).long(), 'indicator': torch.from_numpy(sample['indicator']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

