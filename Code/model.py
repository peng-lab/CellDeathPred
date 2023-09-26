import math

import torch
import torchvision
from torch import nn
from torch.nn import functional as F

#from pytorch_metric_learning import losses, reducers
#from pytorch_metric_learning.distances import CosineSimilarity



class Backbone(nn.Module):
    def __init__(self, backbone, stain, bn_mom = 0.1):
        super().__init__()
         
        if stain == 1:
            stain_ch = 4
        elif stain == 2:
            stain_ch = 3
            #stain_ch = 4 # for channels duplication
        elif stain == 3:
            stain_ch = 4
        else:
            stain_ch = 7
        '''backbone for the architecture. 
            Supported backbones: ResNets, ResNeXts, DenseNets (from torchvision), EfficientNets.'''
        if backbone.startswith('densenet'):
            channels = 96 if backbone == 'densenet161' else 64
            first_conv = nn.Conv2d(stain_ch, channels, 7, 2, 3, bias=True) 
            pretrained_backbone = getattr(torchvision.models, backbone)(pretrained=True)
            self.features = pretrained_backbone.features
            self.features.conv0 = first_conv
            self.features_num = pretrained_backbone.classifier.in_features
        elif backbone.startswith('resnet') or backbone.startswith('resnext'):
            first_conv = nn.Conv2d(stain_ch, 64, 7, 2, 3, bias=False)
            pretrained_backbone = getattr(torchvision.models, backbone)(pretrained=True)
            
            w = pretrained_backbone.conv1.weight
            first_conv.weight = nn.Parameter(torch.stack([torch.mean(w, 1)]*stain_ch, dim=1)) # weights for all channels are same and equal to the mean of pretrained  channels 
            
            self.features = nn.Sequential(
                first_conv,
                #pretrained_backbone.conv1,
                pretrained_backbone.bn1,
                pretrained_backbone.relu,
                pretrained_backbone.maxpool,
                pretrained_backbone.layer1,
                pretrained_backbone.layer2,
                pretrained_backbone.layer3,
                pretrained_backbone.layer4,
                pretrained_backbone.avgpool,
            )
            self.features_num = pretrained_backbone.fc.in_features
        elif backbone.startswith('efficientnet'):
            '''
            from efficientnet_pytorch import EfficientNet
            self.efficientnet = EfficientNet.from_pretrained(backbone)
            first_conv = nn.Conv2d(6, self.efficientnet._conv_stem.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.efficientnet._conv_stem = first_conv
            self.features = self.efficientnet.extract_features
            self.features_num = self.efficientnet._conv_head.out_channels
            '''
            first_conv = nn.Conv2d(stain_ch, 32, 3, 2, 1, bias=False)
            pretrained_backbone = getattr(torchvision.models, backbone)(pretrained=True)
            #self.features = pretrained_backbone.features
            w = pretrained_backbone.features[0][0].weight # shape [32, 4, 3, 3]
            with torch.no_grad():
                for ch in [0,1,2]:
                    first_conv.weight[:,ch,:,:] = nn.Parameter(w[:,ch,:,:])
                avg_ch = torch.mean(w, 1) # avg of three channels
                first_conv.weight[:,3,:,:] = nn.Parameter(avg_ch) # assign forth channel to the avg
            pretrained_backbone.features[0][0] = first_conv
            
            self.features = nn.Sequential(#efficient net architecture
                pretrained_backbone.features[0],
                pretrained_backbone.features[1],
                pretrained_backbone.features[2],
                pretrained_backbone.features[3],
                pretrained_backbone.features[4],
                pretrained_backbone.features[5],
                pretrained_backbone.features[6],
                pretrained_backbone.features[7],
                pretrained_backbone.features[8],
                pretrained_backbone.avgpool,
            )
            self.features_num = pretrained_backbone.classifier[1].in_features
            
        else:
            raise ValueError('wrong backbone')
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_mom
        
    #@torch.cuda.amp.autocast()
    def forward(self, x):
        #x = self.features.conv0(x)
        out_backbone = self.features(x)
        
        out_backbone = torch.squeeze(out_backbone) 

        #z = self.gap(out_backbone)
        
        #z = z.view(z.size(0), -1)
        
        #if self.concat_cell_type:
        #    x = torch.cat([x, s], dim=1)

        #embedding = self.neck(z)
        #print ("out backbone  ", out_backbone.shape)
        return out_backbone

class Embedder(nn.Module):
    def __init__(self, num_features, embedding_size = 512, emb_hidden = None, disable_emb = False, bn_mom = 0.1):
        super().__init__()
        
        #self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        if emb_hidden is None:
            if disable_emb:
                self.neck = nn.Identity()
            else:
                self.neck = nn.Sequential(
                nn.Linear(num_features, embedding_size, bias=True), # add ReLU?
                nn.BatchNorm1d(embedding_size),
        )
        else:
            self.neck = []
            for input_size, output_size in zip([num_features] + emb_hidden, emb_hidden):
                self.neck.extend([
                    nn.Linear(input_size, output_size, bias=True),
                    nn.ReLU(), # inplace=True? 
                    nn.BatchNorm1d(output_size), #after or before activation?
                ])
            self.neck.append(nn.Linear(emb_hidden[-1], embedding_size))
            self.neck = nn.Sequential(*self.neck)
        
        ''' 
        self.neck = nn.Sequential(
            #nn.BatchNorm1d(num_features),
            nn.Linear(num_features, num_features, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, embedding_size, bias=True),
            nn.BatchNorm1d(embedding_size),
        )
        '''
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_mom
        
    def forward(self, out_backbone):
        #x = self.features(out_backbone)
        
        #x = F.adaptive_avg_pool2d(out_backbone, (1, 1))
        #x = nn.AdaptiveAvgPool2d(out_backbone, (1, 1))
        #x = self.gap(out_backbone)
        
        #x = x.view(x.size(0), -1)
        #if self.concat_cell_type:
        #    x = torch.cat([x, s], dim=1)

        #print ("input to embedder ", out_backbone.shape)
        # wo squeeze torch.Size([150, 512, 1, 1]) = (bs*crops, num_features, 1, 1)
        # If the tensor has a batch dimension of size 1, then squeeze(input) will also remove the batch dimension, which can lead to unexpected errors.
        #out_backbone = torch.squeeze(out_backbone) 
        embedding = self.neck(out_backbone) 
        return embedding

class Classifier(nn.Module):
    def __init__(self, embedding_size, classes, head_hidden = None, bn_mom = 0.1):
        super().__init__()
        
        # hidden layers sizes in the head. Defaults to absence of hidden layers
        if head_hidden is None:
            self.head = nn.Linear(embedding_size, classes)
        else:
            self.head = []
            for input_size, output_size in zip([embedding_size] + head_hidden, head_hidden):
                self.head.extend([
                    nn.Linear(input_size, output_size, bias=True), # bias = False?
                    nn.BatchNorm1d(output_size),
                    nn.ReLU(),
                ])
            self.head.append(nn.Linear(head_hidden[-1], classes))
            self.head = nn.Sequential(*self.head)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_mom
    
    #@torch.cuda.amp.autocast()
    def forward(self, embeddings):
        out = self.head(embeddings)
        return out
