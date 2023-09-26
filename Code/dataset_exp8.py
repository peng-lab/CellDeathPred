import cv2
import logging
import math
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from itertools import chain
from operator import itemgetter
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from PIL import Image

from skimage import exposure

#python_code = __import__('kaggle-rcic-1st')
#import python_code

# import precomputed as P
import ipdb

def cv2_clipped_zoom(img, zoom_factor=0):
    #from https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    ------
    Args:
        img : ndarray with shape (channel, height,width)
            Image array
        zoom_factor : float
            amount of zoom as a ratio [0 to Inf). Default 0.
    ------
    Returns:
        result: ndarray
           numpy ndarray of the same shape of the input img zoomed by the specified factor.          
    """
    if zoom_factor == 0:
        return img


    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    
    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]
    
    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)
    
    result = cv2.resize(cropped_img, (resize_width, resize_height), interpolation = cv2.INTER_LINEAR)
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result


def tta(args, images):
    """Augment all images in a batch and return list of augmented batches"""

    ret = []
    n1 = math.ceil(args.tta ** 0.5)
    n2 = math.ceil(args.tta / n1)
    k = 0
    for i in range(n1):
        for j in range(n2):
            if k >= args.tta:
                break

            dw = round(args.tta_size * images.size(2))
            dh = round(args.tta_size * images.size(3))
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


def worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 10 ** 9) + worker_id)


def train_transform2(image):
    #ipdb.set_trace()
    # image.shape[0] = number of channels
    a, b = np.random.normal(1, 0.1, (image.shape[0], 1, 1)), np.random.normal(
        0, 0.1, (image.shape[0], 1, 1))
    a, b = torch.tensor(a, dtype=torch.float32), torch.tensor(
        b, dtype=torch.float32)
    return image * a + b


def get_train_val_loader(args, train_transform, val_transform = None, filter_fields = False, channels = [1,2,3,4], predict=False):
    def train_transform1(image):
        #ipdb.set_trace()
        if random.random() < 0.5:
            image = image[:, ::-1, :]
        if random.random() < 0.5:
            image = image[::-1, :, :]
        if random.random() < 0.5:
            image = image.transpose([1, 0, 2])
        image = np.ascontiguousarray(image)

        if args.scale_aug != 1:
            size_x = random.randint(
                round(1360 * args.scale_aug), 1360)  # change to 1024?
            size_y = random.randint(round(1024 * args.scale_aug), 1024)
            x = random.randint(0, 1360 - size_x)
            y = random.randint(0, 1024 - size_y)
            image = image[x:x + size_x, y:y + size_y]
            image = cv2.resize(image, (1360, 1024),
                               interpolation=cv2.INTER_NEAREST)

        return image
    
    def train_transform2(image):
        #ipdb.set_trace()
        # image.shape[0] = number of channels
        a, b = np.random.normal(1, args.pw_aug[0], (image.shape[0], 1, 1)), np.random.normal(
            0, args.pw_aug[1], (image.shape[0], 1, 1))
        a, b = torch.tensor(a, dtype=torch.float32), torch.tensor(
            b, dtype=torch.float32)
        return image * a + b
    
    

    if not predict:
        train_dataset = CellularDatasetNew8(args.data, 'train', transform=train_transform, filter_fields = filter_fields, crop=args.crop, select_plates=args.select_plates, channels=channels, ds_h = args.ds_h, cv_number=args.cv_number, split_seed=args.data_split_seed, choose_range = args.choose_range)
        # ipdb.set_trace()
        # train = DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_data_workers, worker_init_fn=worker_init_fn)

    for i in range(1 if not predict else 2):
        val_dataset = CellularDatasetNew8(args.data, 'val' if i == 0 else 'train', transform=val_transform, crop=args.crop, filter_fields = filter_fields, select_plates=args.select_plates, channels = channels, ds_h = args.ds_h, cv_number=args.cv_number, split_seed=args.data_split_seed, choose_range = args.choose_range)
        # ipdb.set_trace()
        #assert(not set(train_dataset.data).isdisjoint(dataset.data)) == True
        #print (len(set(train_dataset.data).intersection(set(dataset.data))))
        # loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_data_workers, worker_init_fn=worker_init_fn)
        # if i == 0:
        #     val = loader
        # else:
        #     train = loader

    assert len(set(train_dataset.data).intersection(
        set(val_dataset.data))) == 0

    return train_dataset, val_dataset


def get_test_loader(args):
    test_dataset = CellularDatasetNew8(args.data, 'test', stain=args.stain, zone = args.zone, concat_zeroes=args.concat_zeroes, crop=args.crop)
    return DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_data_workers,
                      worker_init_fn=worker_init_fn)

def get_test_dataset(args, filter_fields, channels, test_transform = None):
    test_dataset = CellularDatasetNew8(args.data, 'test', transform=test_transform, filter_fields = filter_fields, crop=args.crop, channels=channels )
    return test_dataset


class CellularDatasetNew8(Dataset):
    treatment_classes = 3

    def __init__(self, root_dir, mode, confocal, magnification, split_seed=1, cv_number=0, transform=None, filter_fields = False, crop=True, select_plates = [1,2,3], channels=[1,2,3,4], ds_h='28', visualize=False, choose_range='range2', return_original=False, zoom20_40=False):

        super().__init__()

        self.root = Path(root_dir)
        self.transform = transform
        self.crop = crop
        self.channels = channels
        self.ds_h = ds_h # 24h or 72h dataset
        self.visualize = visualize
        self.return_original = return_original
        
        self.confocal = confocal
        self.magnification = magnification
        self.zoom20_40 = zoom20_40
        
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        
        #xls = pd.ExcelFile(self.root / 'list_treatment_tingying_experiment_6.xlsx')
        #self.csv = pd.read_excel(xls, sheet_name = f'{self.ds_h}h')
        
        if choose_range == 'range1_high':
            self.csv = pd.read_csv(self.root / '')
        elif choose_range == 'range1' :
            self.csv = pd.read_csv(self.root / '')
        elif choose_range == 'all':
            self.csv = pd.read_csv(self.root / 'df_exp8_all.csv')
        elif choose_range == 'range2' :
            self.csv = pd.read_csv(self.root / 'df_exp8_range2.csv')
        elif choose_range == 'range2_high':
            self.csv = pd.read_csv(self.root / '')
            
        self.data = []

        #healthy_list = ['untreated', 'DMSO']
        #apop_list = ['Stauro', 'Vinblastine', 'Paclitaxel', 'Etoposide', 'Actinomycin D', 'Doxorubicin', 'Niclosamide']
        #ferrop_list = ['RLS3', 'ML162','ML210', 'Erastin','IKE','FINO2','FIN56']

        '''
        #plates_char = select_plates
        #plates_int = [ord(x)-93 for x in plates_char]
        #plates = list(list(zip(plates_char, plates_int)))
        for plate in train_plates:
            for row in self.csv.iterrows():
                r = row[1]
                if (r.Row not in [1, 16]) and (r['Plate Column'] not in [1, 24]):
                    for field in range(1, 11):  # taking all 10 fields!
                    #for field in range(3, 7): # 4 fields  
                        if r.Treatment in healthy_list:
                            drug_encoding = 2
                        elif r.Treatment in apop_list:
                            drug_encoding = 0
                        elif r.Treatment in ferrop_list:
                            drug_encoding = 1
                        #if r.Cell_Death == 'Unknown':
                        #    drug_encoding = 3
                        
                        self.data.append((int(r.Row), int(r['Plate Column']), r.Treatment, r['Concentration [µM]'], field, plate, drug_encoding))
                
        '''
        if self.magnification == '20x':
            num_fields = 3
        else:
            num_fields = 10 # for 40x
            
        for row in self.csv.iterrows():
            r = row[1]
            if r.plate in select_plates:
                for field in range(1, num_fields + 1):  
                #for field in range(3, 7): # 4 fields  
                    if r.drug == 'healthy':
                        drug_encoding = 2
                    elif r.drug == 'apoptosis':
                        drug_encoding = 0
                    elif r.drug == 'ferroptosis':
                        drug_encoding = 1
                    
                    self.data.append((int(r['Row_New']), int(r['Col']), r.Treatment, r['concentr'],r['normalized_atp'], field, r.plate, r.drug, drug_encoding))

        '''
        if filter_fields: # by variance
            df_data = pd.DataFrame(self.data, columns=['stain', 'plate', 'Wellname', 'row', 'column','field', 'drug', 'concentration', 'drug_type', 'drug_type_encode'])
            #print (df_var)
            #print (df_data)
            df_data['var'] = df_var['var']
            self.data = list(df_data[df_data['var'] >= 2.0].drop(columns=['var']).itertuples(index=False, name=None)) # to list of tuples
            
        '''
        # randomly pick 10% of len(self.data) indices for validation
        #validation_indices = [random.randrange(0, len(self.data)) for i in range(0.1 * len(self.data) )]
        # print(foo[random_index])
        # ipdb.set_trace()
        if mode != 'test':
            # for reproducibilty of splitting
            state = random.Random(split_seed)
            if cv_number != -1:
                # ipdb.set_trace()
                val = state.sample(self.data, int(0.2 * len(self.data)))
            else:
                val = []

            #all = self.data.copy()
            #tr = sorted(set(all) - set(val))
            tr = [item for item in self.data if item not in val]
            #print (len(set(tr).intersection(set(val))))
            if mode == 'train':
                #logging.info('Train dataset: {}'.format(sorted(tr)))
                # ipdb.set_trace()
                #self.data = list(filter(lambda d: d in tr, self.data))
                self.data = tr
            elif mode == 'val':
                #logging.info('Val dataset: {}'.format(val))
                # ipdb.set_trace()
                #self.data = list(filter(lambda d: d in val, self.data))
                self.data = val

                #validat_set = random.sample(self.data, int(0.1 * len(self.data)))
                #train_set = [item for item in self.data if item not in validat_set]
            else:
                assert 0
        
        
        # for sampler;
        self.plates = [self.data[i][6] for i in range(len(self.data))] 
        self.drug_type = [self.data[i][-1] for i in range(len(self.data))] # r.drug_type_encode
        self.targets = np.array(list(zip(self.plates, self.drug_type)))
        '''
        assert len(set(self.data)) == len(self.data)
        assert len(set(all_data)) == len(all_data)
            '''

        self.filter()

        logging.info('{} dataset size: data: {}'.format(mode, len(self.data)))

    def filter(self, func=None):
        """
        Filter dataset by given function. If function is not specified, it will clear current filter
        :param func: func((index, (experiment, plate, well, site, cell_type, sirna or None))) -> bool
        """
        if func is None:
            self.data_indices = None
        else:
            self.data_indices = list(
                filter(lambda i: func(i, self.data[i]), range(len(self.data))))

    def __len__(self):
        return len(self.data_indices if self.data_indices is not None else self.data)

    def __getitem__(self, i):
        i = self.data_indices[i] if self.data_indices is not None else i
        d = self.data[i]

       # i {Unnamed: 0-> Unnamed:1 	 row 	column 	Wellname 	plate 	drug 	concentration 	stain 	drug,concentration 	drug,concentration_encoded}
        #image_metadata = self.csv.iloc[i]
        # e.g. 014011-4-001001003: Row14Column11-ImageField4-Channel3
        # 002001-1-001001001
        # img_name = os.path.join(self.root,
        #                        self.landmarks_frame.iloc[idx, 0])

        #channels_stain1 = [1, 2, 3, 4]
        #channels_stain2 = [1, 2, 3]

        #d = int(r['Row_New']), int(r['Col']), r.Treatment, r['concentr'],r['normalized_atp'], field, r.plate, r.drug, drug_encoding))
        #stain = d[0]
        row = d[0]
        column = d[1]
        drug = d[2]
        concentration = d[3]
        atp = d[4]
        field = d[5]
        plate = d[6]
        label = d[-1]

        # to account for single digit positions in the names
        if row < 10:
            row = '0' + str(row)
        if column < 10:
            column = '0' + str(column)
        if field < 10:
            field = '0' + str(field)
        
        # confocal/non-confocal
        if self.confocal: 
            image_folder = self.root / f'2022-07-14_Tingying_Exp8_plate0{plate}_{self.magnification}_confocal' / 'Images'
        else:
            image_folder = self.root / f'2022-07-14_Tingying_Exp8_plate0{plate}_{self.magnification}_non-confocal' / 'Images'
        
        imgs = []
       
        for channel in self.channels:
            path = image_folder / f'r{row}c{column}f{field}p01-ch{channel}sk1fk1fl1.tiff'
            #print (path)
            if self.visualize:
                imgs.append(np.array(Image.open(path).convert('L')).astype(np.float32) / 255.0) # gray_img 0,1 
            else:
                #imgs.append(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)) # astype is important!
                #imgs.append(np.array(Image.open(path).convert('L')).astype(np.float32) / 255.0) # gray_img 0,1 
                im = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
                #imgs.append(exposure.equalize_adapthist(im, clip_limit=0.01, nbins=65536).astype(np.float32))
                imgs.append(exposure.equalize_hist(im).astype(np.float32))
        
        image = np.stack(imgs, axis=-1)

        #image = np.array(image).transpose(1, 0, 2)

        # print (image.shape) # (1024, 1360, 3)
        # logging.info('shape{}'.format(image.shape))
        
        #ipdb.set_trace()
        image = np.transpose(image, (1, 0, 2))  # (1360, 1024, 3)
        #if self.transform is not None:
        #    image = self.transform[0](image)
        
        #if self.zoom20_40:
        #    image = cv2_clipped_zoom(image, zoom_factor=2) # center zoom

        image = F.to_tensor(image)  # changes shape to (4, 1360, 1024)
        
        #'''
        if self.zoom20_40:
            #transform_zoom = transforms.RandomResizedCrop(size = (1360,1024), scale=(0.5, 0.5))
            #image = transform_zoom(image)
            transform2 = transforms.CenterCrop(size = (1360 // 2,1024 // 2))
            transform3 = transforms.Resize(size = (1360,1024))
            image = transform3(transform2(image))
        #''
        '''
        # to mean 0 & std 1 (standardization)
        mean = torch.mean(image, dim=(1, 2))
        std = torch.std(image, dim=(1, 2))
        image = (image - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1)
        #print(torch.min(image, dim=0), torch.max(image, dim=0))
        '''
        # try for resnet requirement
        #image = F.resize(image, (512,512))
        
        # logging.info('shape{}'.format(image.shape))
        
        if self.return_original:
            image_original = image
        
        #if self.transform is not None:
        if self.mode == 'train':
            if self.crop:
                # also try TenCrop?
                crops = F.five_crop(image, size=(512, 512))# tuple of 5 crops with 4,512,512
                crops_transf = []
                for img in crops: # applying transfs to each crop separately
                    if self.transform is not None:
                        if random.random() < 0.5:
                            img = img.permute(0, 2, 1) # 90 degrees rotation
                        '''
                        # to mean 0 & std 1 (standardization)
                        mean = torch.mean(img, dim=(1, 2))
                        std = torch.std(img, dim=(1, 2))
                        img = (img - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1)
                        '''
                        img = self.transform(img)
                        img = train_transform2(img) # gaussian noise
                    
                    if len(self.channels) == 3:
                        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # rssnet
                    else:
                        img = F.normalize(img, mean=[0.578, 0.533, 0.509, 0.599], std=[0.204, 0.249, 0.281, 0.186]) # pl 1,2
                    crops_transf.append(img)
                    #print(torch.min(img, dim=0), torch.max(image, dim=0))
                image = torch.stack([crop for crop in crops_transf]) # 5,4,512,512
            else:
                #if random.random() < 0.5:
                #    image = image.permute(0, 2, 1) # 90 degrees rotation
                #image = train_transform2(image) # gaussian noise
                image = self.transform(image)
        
        if self.mode == 'val' or self.mode == 'test':
            if self.crop:
                crops = F.five_crop(image, size=(512, 512))# tuple of 5 crops with 4,512,512
                crops_transf = []
                for img in crops: # applying transfs to each crop separately
                    if random.random() < 0.5:
                        img = img.permute(0, 2, 1) # 90 degrees rotation
                    '''   
                    # to mean 0 & std 1 (standardization)
                    mean = torch.mean(img, dim=(1, 2))
                    std = torch.std(img, dim=(1, 2))
                    img = (img - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1)
                    '''   
                    img = self.transform(img)
                    img = train_transform2(img) # gaussian noise
                    if len(self.channels) == 3:
                        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # resnet
                    else:
                        img = F.normalize(img, mean=[0.578, 0.533, 0.509, 0.599], std=[0.204, 0.249, 0.281, 0.186]) # pl 1,2
                    crops_transf.append(img)
                image = torch.stack([crop for crop in crops_transf]) # 5,4,512,512
            else:
                #if random.random() < 0.5:
                 #   image = image.permute(0, 2, 1) # 90 degrees rotation
                #image = train_transform2(image) # gaussian noise
                image = self.transform(image)
        
        ''' 
        filter crops based on variance
        if self.transform is not None:
            if self.mode == 'train':
                # also try TenCrop?
                crops = F.five_crop(image, size=(512, 512))# tuple of 5 crops with 4,512,512
                crops_transf = []
                for img in crops: # applying transfs to each crop separately
                    if torch.var(img) >= 2.0:
                        if random.random() < 0.5:
                            img = img.permute(0, 2, 1) # 90 degrees rotation
                        img = train_transform2(img) # gaussian noise
                        img = self.transform(img)
                        crops_transf.append(img)
                    if not crops_transf: # if all var(crops) of an image are < 2.0 
                        return None
                image = torch.stack([crop for crop in crops_transf]) # 5,4,512,512
            
        if self.mode == 'val' or self.mode == 'test':
            crops = F.five_crop(image, size=(512, 512))# tuple of 5 crops with 4,512,512
            crops_transf = []
            for img in crops: # applying transfs to each crop separately
                if torch.var(img) >= 2.0:
                    if random.random() < 0.5:
                        img = img.permute(0, 2, 1) # 90 degrees rotation
                    #img = train_transform2(img) # gaussian noise
                    img = self.transform(img)
                    crops_transf.append(img)
                if not crops_transf: # if all var(crops) of an image are < 2.0 
                    return None
            image = torch.stack([crop for crop in crops_transf]) # 5,4,512,512
            #image = torch.stack([crop for crop in crops]) # 5,4,512,512
        ''' 
        #image = image.permute(0, 2, 1)

        # cell_type = nn.functional.one_hot(torch.tensor(self.cell_types.index(d[-2]), dtype=torch.long),
            # len(self.cell_types)).float()

        r = [image, torch.tensor(int(field), dtype=torch.long), torch.tensor(i, dtype=torch.long), torch.tensor(int(row), dtype=torch.long), torch.tensor(int(column), dtype=torch.long), torch.tensor(plate, dtype=torch.long)]
        '''
        r = [image, torch.tensor(self.stain, dtype=torch.long).repeat(image.shape[0]), torch.tensor(
            field, dtype=torch.long).repeat(image.shape[0]), torch.tensor(i, dtype=torch.long).repeat(image.shape[0]), torch.tensor(plate, dtype=torch.long).repeat(image.shape[0])]
        '''
        # if self.mode != 'test':
        #r.append(torch.tensor(label, dtype=torch.long))
        
        if self.return_original:
            r[1:1] = [image_original]
            #r.insert(index = 1, obj = image_original)
            
        r.append(torch.tensor(label, dtype=torch.long))
        # image, stain, field, index, label
        return tuple(r)
