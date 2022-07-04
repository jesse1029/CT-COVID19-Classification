""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2019, Ross Wightman
"""
import torch.utils.data as data
import os
import torch
import logging

from PIL import Image

from .parsers import create_parser

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            parser=None,
            class_map=None,
            load_bytes=False,
            transform=None,
            target_transform=None,
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            parser=None,
            split='train',
            is_training=False,
            batch_size=None,
            repeats=0,
            download=False,
            transform=None,
            target_transform=None,
    ):
        assert parser is not None
        if isinstance(parser, str):
            self.parser = create_parser(
                parser, root=root, split=split, is_training=is_training,
                batch_size=batch_size, repeats=repeats, download=download)
        else:
            self.parser = parser
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.parser:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.parser, '__len__'):
            return len(self.parser)
        else:
            return 0

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)

###############################################################################

import numpy as np
import pandas as pd
import random
from tqdm import tqdm

def array_convert(string):

    A = string.replace('[', '').replace(']', '').replace('\n', '').split(' ')
    new_array = np.array([float(x) for x in A if len(x) > 0])

    return new_array

class COVID_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, csv_path, data_split, transform = None):
        ###
        # csv_path = 'df_224_v2.csv'
        # data_split = 'train'
        # csv_path = 'test_224_embed.pkl'
        ###
        if data_split == 'test':
            
            dataset = pd.read_pickle(csv_path)
            
            self.embed_info = dataset
            
            classes = [999]
            classes = sorted(list(classes))
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            
            ct_path = np.unique(dataset.iloc[:, 2])
            imgs = []
            for i_scan_dir in tqdm(ct_path):
                temp_df = dataset[dataset['ct_path'] == i_scan_dir]
                imgs.append((i_scan_dir, 999))
            
        elif data_split == 'train' or 'valid': 
            
            df = pd.read_csv(csv_path)
            df['embed'] = df['embed'].apply(array_convert)
            dataset = df[df['split'] == data_split]
            
            self.embed_info = dataset
            
            classes = set(dataset['label'])
            classes = sorted(list(classes))
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            
            ct_path = np.unique(dataset.iloc[:, 4])
            imgs = []
            for i_scan_dir in ct_path:
                temp_df = dataset[dataset['ct_path'] == i_scan_dir]
                imgs.append((i_scan_dir, temp_df.iloc[0, 3]))

        self.classes = classes
        self.class_to_idx = class_to_idx
            
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        img_scan_dir, label = self.imgs[index]
        label = self.class_to_idx[label]
        # img = Image.open(img_name).convert('RGB')
        
        ###
        temp_df = self.embed_info[self.embed_info['ct_path'] == img_scan_dir]
        
        # index = 1
        # img_scan_dir, label = imgs[index]
        # temp_df = embed_info[embed_info['ct_path'] == img_scan_dir]
        
        random.seed(4019)
        if len(temp_df) >= 224:
            
            temp_index = [x for x in range(len(temp_df))]
            target_index = random.sample(temp_index, k = 224)
            
            
        elif len(temp_df) < 224:
            target_index = [x for x in range(len(temp_df))]
            temp = random.choices(target_index, k = 224 - len(target_index))
            target_index += temp
        
        target_index.sort()
        embed = temp_df.iloc[target_index, 1]
        img = np.array([])
        for i_embed in embed:
            img = np.concatenate([img, i_embed])
        img = img.reshape(1, -1, 224)
        
        ### 
        # temp_img = img.copy()
        # img = np.concatenate([img, temp_img], axis = 0)
        # img = np.concatenate([img, temp_img], axis = 0)
        
        ###
        # if self.transform is not None:
        #     img = self.transform(img)
        
        return img, label

    def __len__(self):
        return len(self.imgs)

# ds = COVID_Dataset(csv_path = 'test_224_embed.pkl', data_split = 'test', transform = None)
# from torch.utils.data import DataLoader
# # train_dataloaders = DataLoader(ds, batch_size = 16, shuffle = True, num_workers = 0)
# test_dataloaders = DataLoader(ds, batch_size = 16, shuffle = False, num_workers = 0)

# filenames = [x[0] for x in test_dataloaders.dataset.imgs]

# from tqdm import tqdm
# for (x, y) in tqdm(test_dataloaders):
#     print(f'x:{x.shape}, y:{y}')

# A = [1,2,3,4]
# with open('./covid.csv', 'w') as out_file:
#     for i in A:
#         out_file.write('{0}\n'.format(str(i)))



