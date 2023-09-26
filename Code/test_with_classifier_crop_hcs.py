import math
import random
import os

import tqdm
import numpy as np
import torch
from torch import nn

from pytorch_metric_learning.utils import common_functions as c_f
#from ..utils import common_functions as c_f

from pytorch_metric_learning.testers import GlobalEmbeddingSpaceTester 

import ipdb

#from dataset import CellularDataset
from torch.utils.data import DataLoader

#from ray import tune

@torch.no_grad()
def smooth_label(Y, classes=13, label_smoothing=0):
    nY = nn.functional.one_hot(Y, classes).float()
    nY += label_smoothing / (classes - 1)
    nY[range(Y.size(0)), Y] -= label_smoothing / \
        (classes - 1) + label_smoothing
    return nY


def worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 10 ** 9) + worker_id)


# --lr cosine,1.5e-4,90,6e-5,150,0   
def get_learning_rate(epoch, schedule = 'cosine'):
    #assert len(args.lr[1][1::2]) + 1 == len(args.lr[1][::2])
    for start, end, lr, next_lr in zip([0] + [90.0, 150.0],
                                       [90.0, 150.0] + [130],
                                       [0.00015, 6e-05, 0.0],
                                       [6e-05, 0.0] + [0]):
        if start <= epoch < end:
            if schedule == 'cosine':
                return lr * (math.cos((epoch - start) / (end - start) * math.pi) + 1) / 2
            elif schedule == 'const':
                return lr
            else:
                assert 0
    assert 0
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  


def tta(images, number_of_tta = 8, tta_size = 1):
    """Augment all images in a batch and return list of augmented batches"""

    ret = []
    n1 = math.ceil(number_of_tta ** 0.5)
    n2 = math.ceil(number_of_tta / n1)
    k = 0
    for i in range(n1):
        for j in range(n2):
            if k >= number_of_tta:
                break

            dw = round(tta_size * images.size(2))
            dh = round(tta_size * images.size(3))
            w = i * (images.size(2) - dw) // max(n1 - 1, 1)
            h = j * (images.size(3) - dh) // max(n2 - 1, 1)

            imgs = images[:, :, w:w + dw, h:h + dh]
            if k & 1:
                imgs = imgs.flip(3)
            if k & 2:
                imgs = imgs.flip(2)
            if k & 4:
                imgs = imgs.transpose(2, 3)

            ret.append(nn.functional.interpolate(
                imgs, images.size()[2:], mode='nearest'))
            k += 1

    return ret



class TestWithClassifierCropHCS(GlobalEmbeddingSpaceTester):
    def __init__(self, crop, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.embedding_sizes = embedding_sizes 
        self.crop = crop
    
    def compute_all_embeddings(self, dataloader, trunk_model, embedder_model):
        s, e = 0, 0
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(dataloader)):
                img, label = self.data_and_label_getter(data)
                label = c_f.process_label(label, "all", self.label_mapper)
                #print ("bef label: ", label.shape)
                ''' 
                if not self.avg_embs:
                    label = label.repeat_interleave(5, dim=0)
                    #print ("aft label: ", label.shape)
                    plates, drug_types = torch.tensor(list(zip(*label)))
                    label = drug_types
                '''
                q = self.get_embeddings_for_eval(trunk_model, embedder_model, img)
                #print ("q emb: ", q.shape)
                #print ("aft aft label: ", label.shape)
                if label.dim() == 1:
                    label = label.unsqueeze(1)
                if i == 0:
                    labels = torch.zeros(
                        len(dataloader.dataset),
                        label.size(1),
                        device=self.data_device,
                        dtype=label.dtype,
                    )
                    all_q = torch.zeros(
                        len(dataloader.dataset),
                        q.size(1),
                        device=self.data_device,
                        dtype=q.dtype,
                    )
                e = s + q.size(0)
                all_q[s:e] = q
                labels[s:e] = label
                s = e
        return all_q, labels
    
    def get_embeddings_for_eval(self, trunk_model, embedder_model, input_imgs):
        input_imgs = c_f.to_device(
            input_imgs, device=self.data_device, dtype=self.dtype
        )
        if self.crop:
            bs, ncrops, c, h, w = input_imgs.size()
            #trunk_output = trunk_model(input_imgs.view(-1, c, h, w)).view(bs, ncrops, -1).mean(1) # avg trunk_output
            trunk_output = trunk_model(input_imgs.view(-1, c, h, w))
            if self.use_trunk_output:
                return trunk_output
            
            #if not self.avg_embs:
                #return embedder_model(trunk_output)
            return embedder_model(trunk_output).view(bs,ncrops,-1).mean(1) # avg embs 
        else:
            trunk_output = trunk_model(input_imgs)
            if self.use_trunk_output:
                return trunk_output
            return embedder_model(trunk_output)
    

    
   
