import torch.utils.data
import numpy as np
from imageio import imread
from PIL import Image
from termcolor import colored, cprint
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import datasets
import os.path
import json
import h5py
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode


# from configs.paths import dataroot


## BASE DATASET
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def CreateDataset(opt):
    dataset = None

    # decide resolution later at model
    if opt.dataset_mode == 'snet':
        #from datasets.snet_dataset import ShapeNetDataset
        train_dataset = ShapeNetDataset()
        test_dataset = ShapeNetDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat, res=opt.res)
        test_dataset.initialize(opt, 'test', cat=opt.cat, res=opt.res)

    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    cprint("[*] Dataset has been created: %s" % (train_dataset.name()), 'blue')
    return train_dataset, test_dataset



## DATA LOADER 
def get_data_generator(loader):
    while True:
        for data in loader:
            yield data

def CreateDataLoader(opt):
    train_dataset, test_dataset = CreateDataset(opt)
    print(len(train_dataset), len(test_dataset))
    train_dl = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            sampler=data_sampler(train_dataset, shuffle=True, distributed=opt.distributed),
            drop_last=True,
            )

    test_dl = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            sampler=data_sampler(test_dataset, shuffle=False, distributed=opt.distributed),
            drop_last=False,
            )

    test_dl_for_eval = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=max(int(opt.batch_size // 2), 1),
            sampler=data_sampler(test_dataset, shuffle=False, distributed=opt.distributed),
            drop_last=False,
        )

    return train_dl, test_dl, test_dl_for_eval




# from https://github.com/laughtervv/DISN/blob/master/preprocessing/info.json
class ShapeNetDataset(BaseDataset):

    def initialize(self, opt, phase='train', cat='chair', res=64):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.res = res

        dataroot = opt.dataroot
        # with open(f'{dataroot}/ShapeNet/info.json') as f:
        with open(f'{dataroot}/ShapeNet/filelist/info-shapenet.json') as f:
            self.info = json.load(f)
            
        self.cat_to_id = self.info['cats']
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}
        
        if cat == 'all':
            all_cats = self.info['all_cats']
        else:
            all_cats = [cat]

        self.model_list = []
        self.cats_list = []
        for c in all_cats:
            synset = self.info['cats'][c]
            # with open(f'{dataroot}/ShapeNet/filelists/{synset}_{phase}.lst') as f:
            with open(f'data/ShapeNet/filelist/{phase}_{c}.txt') as f:
                model_list_s = []
                for l in f.readlines():
                    model_id = l.rstrip('\n')
                    #print(model_id)
                    # path = f'{dataroot}/ShapeNet/SDF_v1_64/{synset}/{model_id}/ori_sample_grid.h5'
                    path = f'{dataroot}/ShapeNet/sdf/{model_id}.npy'
                    #print(path)
                    
                    if os.path.exists(path):
                        model_list_s.append(path)
                
                self.model_list += model_list_s
                self.cats_list += [synset] * len(model_list_s)
                print('[*] %d samples for %s (%s).' % (len(model_list_s), self.id_to_cat[synset], synset))

        np.random.default_rng(seed=0).shuffle(self.model_list)
        np.random.default_rng(seed=0).shuffle(self.cats_list)

        self.model_list = self.model_list[:self.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.model_list)), 'yellow')

        self.N = len(self.model_list)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __getitem__(self, index):

        synset = self.cats_list[index]
        sdf_npy_file = self.model_list[index]
        #print(sdf_npy_file)
        
        """ for file in sdf_npy_file:
            print(len(file)) """

        # 修改为读取npy文件
        sdf = np.load(sdf_npy_file).astype(np.float32)
        #print(sdf.shape)
        sdf = torch.Tensor(sdf).view(1, self.res, self.res, self.res)
        
        # sdf = sdf[:, :64, :64, :64]

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        ret = {
            'sdf': sdf,
            'cat_id': synset,
            'cat_str': self.id_to_cat[synset],
            'path': sdf_npy_file,
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'ShapeNetSDFDataset'