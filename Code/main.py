#!/usr/bin/env python3

import itertools
import logging
import math
import pickle
import random
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from pytorch_metric_learning import losses, samplers, miners
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning import testers
import pytorch_metric_learning.utils.logging_presets as logging_presets
import pytorch_metric_learning

import importlib

from model import Backbone, Embedder, Classifier
from train_with_classifier_crop_hcs import TrainWithClassifierCropHCS
from test_with_classifier_crop_hcs import TestWithClassifierCropHCS
from triplet_miner_hcs import TripletMinerHCS

def parse_args():
    def lr_type(x):
        x = x.split(',')
        return x[0], list(map(float, x[1:]))

    def bool_type(x):
        if x.lower() in ['1', 'true']:
            return True
        if x.lower() in ['0', 'false']:
            return False
        raise ValueError()

    parser = ArgumentParser()
    parser.add_argument('-m', '--mode', default='train',
                        choices=('train', 'val', 'test'))

    parser.add_argument('--zone', type=str, default=None,
                        choices=('low', 'high', 'medium'))
    parser.add_argument('--stain', type=int, default=3)
   # parser.add_argument('--stain_for_model', type=int)
    parser.add_argument('--lr_const', type=float, default=1.5e-4)
    parser.add_argument('--metric_loss_weight', type=float, default=0.2)
    parser.add_argument('--classifier_loss_weight', type=float, default=1.0)

    parser.add_argument('--log_path', help='path to log folder')
    parser.add_argument('--tensorboard_path',
                        help='path to tensorboard folder')
    parser.add_argument('--model_save_path', help='path to saved model folder')
    parser.add_argument('--exper_name', help='name of the experiment')

    parser.add_argument('--ds', type=str, help='choose dataset')
    parser.add_argument('--ds_h', type=str, help='choose hour of new dataset')
   
    parser.add_argument('--choose_range', type=str, help='choose a medium range dataset')
    
    parser.add_argument('--miner_margin', type=float, default=0.2)
    parser.add_argument('--samples_per_class', type=int)

    parser.add_argument('--triplet_margin', type=float, default=0.1)

    parser.add_argument('--crop', type=bool_type, default=False)
    parser.add_argument('--avg_embs', type=bool_type, default=True,
                        help='avg embs of a cropped image')
    parser.add_argument('--my_miner', type=bool_type, default=False)

    parser.add_argument('--bn_mom', type=float, default=0.1)
    parser.add_argument('--emb_hidden', type=lambda x: None if not x else list(map(int, x.split(','))),
                        help='hidden layers sizes in the embedder. Defaults to absence of hidden layers')
    parser.add_argument('--disable_emb', type=bool_type, default=False)

    parser.add_argument('--lr_scheduler', type=bool_type)

    parser.add_argument('--train_plates', type=lambda x: None if not x else list(map(int, x.split(','))),
                        help='choose two plates to train on and third will be the test plate; the order is important!')

    parser.add_argument('--select_plates', type=lambda x: None if not x else list(map(int, x.split(','))),
                        help='choose plates to train on; the order is important!')

    parser.add_argument('--load_model', type=str,
                        help='choose pretrained trunk & embedder models')
    parser.add_argument('--checkpoint', type=int)

    parser.add_argument('--backbone', default='resnet18',
                        help='backbone for the architecture. '
                        'Supported backbones: ResNets, ResNeXts, DenseNets (from torchvision), EfficientNets. '
                        'For DenseNets, add prefix "mem-" for memory efficient version')
    parser.add_argument('--head_hidden', type=lambda x: None if not x else list(map(int, x.split(','))),
                        help='hidden layers sizes in the head. Defaults to absence of hidden layers')
    parser.add_argument('--concat-cell-type', type=bool_type, default=True)

    parser.add_argument('--concat-zeroes', type=bool_type, default=False)

    parser.add_argument('--metric-loss-coeff', type=float, default=0)
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--wd', '--weight-decay', type=float, default=1e-3)
    parser.add_argument('--label-smoothing', '--ls', type=float, default=0)
    parser.add_argument('--mixup', type=float, default=0,
                        help='alpha parameter for mixup. 0 means no mixup')
    parser.add_argument('--cutmix', type=float, default=1,
                        help='parameter for beta distribution. 0 means no cutmix')

    parser.add_argument('--classes', type=int, default=13,
                        help='number of classes predicting by the network')
    parser.add_argument('--fp16', type=bool_type, default=True,
                        help='mixed precision training/inference')
    parser.add_argument('--disp-batches', type=int, default=50,
                        help='frequency (in iterations) of printing statistics of training / inference '
                        '(e.g. accuracy, loss, speed)')

    parser.add_argument('--tta', type=int,
                        help='number of TTAs. Flips, 90 degrees rotations and resized crops (for --tta-size != 1) are applied')
    parser.add_argument('--tta-size', type=float, default=1,
                        help='crop percentage for TTA')

    parser.add_argument('--save',
                        help='path for the checkpoint with best accuracy. '
                        'Checkpoint for each epoch will be saved with suffix .<number of epoch>')
    parser.add_argument('--load',
                        help='path to the checkpoint which will be loaded for inference or fine-tuning')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--pred-suffix', default='',
                        help='suffix for prediction output. '
                        'Predictions output will be stored in <loaded checkpoint path>.output<pred suffix>')

    parser.add_argument('--pw-aug', type=lambda x: tuple(map(float, x.split(','))), default=(0.1, 0.1),
                        help='pixel-wise augmentation in format (scale std, bias std). scale will be sampled from N(1, scale_std) '
                        'and bias from N(0, bias_std) for each channel independently')
    parser.add_argument('--scale-aug', type=float, default=0.5,
                        help='zoom augmentation. Scale will be sampled from uniform(scale, 1). '
                        'Scale is a scale for edge (preserving aspect)')
    parser.add_argument('--all-controls-train', type=bool_type, default=True,
                        help='train using all control images (also these from the test set)')
    parser.add_argument('--data-normalization', choices=('global', 'experiment', 'sample'), default='sample',
                        help='image normalization type: '
                        'global -- use statistics from entire dataset, '
                        'experiment -- use statistics from experiment, '
                        'sample -- use mean and std calculated on given example (after normalization)')
    parser.add_argument('--data', type=Path, default=Path('/lustre/groups/peng/datasets/kenji/'),
                        help='path to the data root. It assumes format like in Kaggle with unpacked archives')
    parser.add_argument('--cv-number', type=int, default=0, choices=(-1, 0, 1, 2, 3, 4, 5),
                        help='number of fold in 6-fold split. '
                        'For number of given cell type experiment in certain fold see dataset.py file. '
                        '-1 means not using validation set (training on all data)')
    parser.add_argument('--data-split-seed', type=int, default=0,
                        help='seed for splitting experiments for folds')
    parser.add_argument('--num-data-workers', type=int, default=6,
                        help='number of data loader workers')
    parser.add_argument('--seed', type=int,
                        help='global seed (for weight initialization, data sampling, etc.). '
                        'If not specified it will be randomized (and printed on the log)')

    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('--gradient-accumulation', type=int, default=2,
                        help='number of iterations for gradient accumulation')
    parser.add_argument('-e', '--epochs', type=int, default=90)

    args = parser.parse_args()

    '''
    if args.mode == 'train':
        assert args.save is not None
    if args.mode == 'val':
        assert args.save is None
    if args.mode == 'predict':
        assert args.load is not None
        assert args.save is None
    '''
    if args.seed is None:
        args.seed = random.randint(0, 10 ** 9)

    return args


def collate_fn(batch):
    # how batch are formed; len(batch)=bs
    batch = list(filter(lambda x: x is not None, batch))
    '''
    imgs = []
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    l5 = []
    for b in batch: # batch is a list
        imgs.append(b[0])
        l1.append(b[1])
        l2.append(b[2])
        l3.append(b[3])
        l4.append(b[4])
        l5.append(b[5])
    img = torch.cat(imgs) # (5+4+...),ch,h,w
    label1 = torch.cat(l1)
    label2 = torch.cat(l2)
    label3 = torch.cat(l3)
    label4 = torch.cat(l4)
    label5 = torch.cat(l5)
    #label = torch.cat(labels)
    return img, label1,label2,label3,label4,label5,
    '''
    return torch.utils.data.dataloader.default_collate(batch)


def my_data_and_label_getter(output):
    '''A function that takes the output of your dataset's __getitem__ function, and
     returns a tuple of (data, labels).
     If None, then it is assumed that __getitem__ returns (data, labels).
     '''
    #print (len(output))
    #print (output[0].shape)
    #print (output[1].shape)
    # labels = (plate, drug_type)
    X, labels = output[0], (output[4], output[-1])
    return X, labels


def my_data_and_label_getter_24h(output):
    # labels = (plate, drug_type)
    X, labels = output[0], (output[5], output[-1])
    return X, labels


def subset_fields(res_list, num_fields=8):
    res = []
    for i in range(0, len(res_list) - 1, num_fields):
        res.append([res_list[i], res_list[i + 1]])
    return res


def setup_logging(args):
    head = '{asctime}:{levelname}: {message}'
    handlers = [logging.StreamHandler(sys.stderr)]
    if args.mode == 'train':
        handlers.append(logging.FileHandler(args.save + '.log', mode='w'))
    if args.mode == 'predict':
        handlers.append(logging.FileHandler(
            args.load + '.output.log', mode='w'))
    logging.basicConfig(level=logging.DEBUG, format=head,
                        style='{', handlers=handlers)
    logging.info('Start with arguments {}'.format(args))


def setup_determinism(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def main(args):
    trunk = Backbone(backbone=args.backbone,
                     stain=args.stain, bn_mom=args.bn_mom)

    embedder = Embedder(num_features=trunk.features_num,
                        embedding_size=args.embedding_size,
                        bn_mom=args.bn_mom,
                        emb_hidden=args.emb_hidden,
                        disable_emb=args.disable_emb)

    classifier = Classifier(embedding_size=args.embedding_size,
                            classes=args.classes,
                            head_hidden=args.head_hidden,
                            bn_mom=args.bn_mom)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        trunk = nn.DataParallel(trunk)
        embedder = nn.DataParallel(embedder)
        classifier = nn.DataParallel(classifier)

    # option to load pretrained a model with 3 channels
    '''
    path_models = args.load_model
    best_checkpoint = args.checkpoint
    print('pretrained model', args.load_model)

    checkpoint_trunk = torch.load(f"./saved_models_{path_models}/trunk_best{best_checkpoint}.pth")
    checkpoint_embedder = torch.load(f"./saved_models_{path_models}/embedder_best{best_checkpoint}.pth")
    #checkpoint_classifier = torch.load(f"./{path_models}/classifier_best{best_checkpoint}.pth")

    trunk.module.load_state_dict(checkpoint_trunk)
    embedder.module.load_state_dict(checkpoint_embedder)
    #classifier.module.load_state_dict(checkpoint_classifier)
   
    first_conv = nn.Conv2d(4, 64, 7, 2, 3, bias=False) # 4 channels
    w = trunk.module.features[0].weight # shape [64, 4, 7, 7]
    with torch.no_grad():
        for ch in [0,1,2]:
            first_conv.weight[:,ch,:,:] = nn.Parameter(w[:,ch,:,:])
        avg_ch = torch.mean(w, 1) # avg of three channels
        first_conv.weight[:,3,:,:] = nn.Parameter(avg_ch) # assign forth channel to the avg
    trunk.module.features[0] = first_conv
    '''
    
     # option to load pretrained a model with 4 channels
    #'''
    path_models = args.load_model
    best_checkpoint = args.checkpoint
    print('pretrained model', args.load_model)

    checkpoint_trunk = torch.load(f"./output_models_of_Aidin/saved_models_{path_models}/trunk_best{best_checkpoint}.pth")
    checkpoint_embedder = torch.load(f"./output_models_of_Aidin/saved_models_{path_models}/embedder_best{best_checkpoint}.pth")
    checkpoint_classifier = torch.load(f"./output_models_of_Aidin/saved_models_{path_models}/classifier_best{best_checkpoint}.pth")

    trunk.module.load_state_dict(checkpoint_trunk)
    embedder.module.load_state_dict(checkpoint_embedder)
    classifier.module.load_state_dict(checkpoint_classifier)
    #trunk.load_state_dict(checkpoint_trunk)
    #embedder.load_state_dict(checkpoint_embedder)
    #classifier.load_state_dict(checkpoint_classifier)
    #'''
    
    trunk.to(device)
    embedder.to(device)
    classifier.to(device)

    # Set optimizers
    trunk_optimizer = torch.optim.Adam(
        trunk.parameters(), lr=args.lr_const, weight_decay=args.wd)
    classifier_optimizer = torch.optim.Adam(
        classifier.parameters(), lr=args.lr_const, weight_decay=args.wd)  # try other opts (e.g. adamax)
    if not args.disable_emb:
        embedder_optimizer = torch.optim.Adam(
            embedder.parameters(), lr=args.lr_const, weight_decay=args.wd)
        optimizers = {"trunk_optimizer": trunk_optimizer,
                      "embedder_optimizer": embedder_optimizer, "classifier_optimizer": classifier_optimizer}
    else:
        optimizers = {"trunk_optimizer": trunk_optimizer,
                      "classifier_optimizer": classifier_optimizer}
    #optimizers = {"trunk_optimizer": trunk_optimizer, "classifier_optimizer": classifier_optimizer, "metric_loss_optimizer": loss_optimizer}

    # Set the image transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        # transforms.RandomRotation([0,90]),
        # transforms.RandomResizedCrop(size=512)])
        #transforms.RandomApply(transforms.GaussianBlur((55,55), sigma=(0.1,2.0)), p=0.3),
    ])
    val_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5)])

    # actually returns two CellularDataset types not dataloaders. Already augmented with transformations.
    dataset = importlib.import_module(args.ds)
    train_dataset, val_dataset = dataset.get_train_val_loader(args, train_transform=train_transform, val_transform=val_transform, filter_fields = True)
    #train_dataset, val_dataset = dataset.get_train_val_loader(args, train_transform=train_transform, val_transform=val_transform, filter_fields=False, channels = [1,2,3,4])

    print(type(train_dataset))
    # Set the metric loss function
    loss_func = losses.SupConLoss(temperature=0.1)
    #loss_func = losses.ArcFaceLoss(num_classes = args.classes, embedding_size = args.embedding_size)
    # loss_func.to(device) # problem here with NaN?
    #loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
    #loss_optimizer = torch.optim.Adam(loss_func.parameters(), lr=1e-5)

    # loss_func = losses.CircleLoss() # make an option for loss f. in args
    '''
    loss_func = losses.TripletMarginLoss(margin=args.triplet_margin, # need to optimize this parameter
                                        swap=False,
                                        smooth_loss=False,
                                        triplets_per_anchor='all')
    # Set the classification loss:
    '''
    '''
    num_points_drug = train_dataset.csv.groupby('drug_encoded').count().iloc[:, 0].tolist()
    print ("number of points per drug", num_points_drug)
    class_weights = torch.FloatTensor(np.sum(num_points_drug) / num_points_drug).to(device)
    print ("class weights", class_weights)
    classification_loss = torch.nn.CrossEntropyLoss(weight = class_weights)
    '''
    classification_loss = torch.nn.CrossEntropyLoss()

    #print ('targets', train_dataset.targets)
    #print ('targets shape', train_dataset.targets.shape)

    # Set the dataloader sampler
    # sampler = samplers.MPerClassSampler(labels = train_dataset.plates, m = 15, batch_size=30, length_before_new_iter=len(train_dataset)) # maybe need to change bn_mom because of batch size * crops
    sampler = samplers.HierarchicalSampler(
        labels=train_dataset.targets,
        batch_size=args.batch_size,
        samples_per_class=args.samples_per_class,  # Y
        batches_per_super_tuple=127, # (24h: 4928/120 = 42) (dataset stain2 = 15)
        #super_classes_per_batch=2,  # X
        super_classes_per_batch=7,  # X number of plates
        inner_label=1,  # class (drug_type)
        outer_label=0  # super class (plate)
    )
    #print ('sampler super pairs', sampler.super_pairs)
    #print ('sampler image lists', sampler.super_image_lists)
    # print ('sampler batches', sampler.batches) # returns a batch of indices at a time

    #print ('sampler', sampler.labels_to_indices)

    # Set the mining function
    # miner = miners.TripletMarginMiner(margin=args.miner_margin, type_of_triplets="hard") # tune margin
    miner = TripletMinerHCS(
        collect_stats=True, margin=args.miner_margin, type_of_triplets="hard")  # tune margin

    # Package the above stuff into dictionaries.
    models = {"trunk": trunk, "embedder": embedder, "classifier": classifier}
    #models = {"trunk": trunk, "classifier": classifier}

    loss_funcs = {"metric_loss": loss_func,
                  "classifier_loss": classification_loss}
    #loss_funcs = {"metric_loss": loss_func}
    if args.my_miner:
        mining_funcs = {"tuple_miner": miner}
    else:
        mining_funcs = {}

    metric_loss_weight = args.metric_loss_weight
    classifier_loss_weight = args.classifier_loss_weight
    # We can specify loss weights if we want to. This is optional
    loss_weights = {"metric_loss": metric_loss_weight,
                    "classifier_loss": classifier_loss_weight}

    accuracy_calculator = AccuracyCalculator(include=(),
                                             exclude=(),
                                             avg_of_avgs=False,
                                             k=None,
                                             label_comparison_fn=None)

    record_keeper, _, _ = logging_presets.get_record_keeper(
        args.log_path, args.tensorboard_path)
    hooks = logging_presets.get_hook_container(
        record_keeper,
        save_models=True,
        #record_group_name_prefix = args.exper_name
    )  # change primary metric?
    dataset_dict = {"train": train_dataset, "val": val_dataset}
    model_folder = args.model_save_path

    '''
    def end_of_testing_hook(tester):
        print(tester.all_accuracies)
    '''

    #splits_to_eval = [('val', ['val', 'train'])]

    if args.ds in ['dataset_24h', 'dataset_exp6', 'dataset_exp5', 'dataset_exp7']:
        data_and_label_getter = my_data_and_label_getter_24h
    else:
        data_and_label_getter = my_data_and_label_getter

    # Create the tester
    tester = TestWithClassifierCropHCS(normalize_embeddings=True, use_trunk_output=False, batch_size=args.batch_size, dataloader_num_workers=args.num_data_workers,
                                       #data_and_label_getter = my_data_and_label_getter_24h,
                                       data_and_label_getter=data_and_label_getter,
                                       accuracy_calculator=accuracy_calculator,
                                       end_of_testing_hook=hooks.end_of_testing_hook,
                                       label_hierarchy_level=1,
                                       crop=args.crop,
                                       #avg_embs = args.avg_embs,
                                       )

    # Or if your model is composed of a trunk + embedder
    #all_accuracies = tester.test(dataset_dict, epoch, trunk, embedder)

    #end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder, splits_to_eval = splits_to_eval)
    end_of_epoch_hook = hooks.end_of_epoch_hook(
        tester, dataset_dict, model_folder, patience = 50)  # wo splits_to_eval;make an option

    '''
    T_0 = 90
    eta_min = 1e-5
    verbose = True
    trunk_lr = torch.optim.lr_scheduler.CosineAnnealingLR(trunk_optimizer,
                                            T_max = T_0,
                                            eta_min=eta_min,
                                            verbose=verbose)

    classifier_lr = torch.optim.lr_scheduler.CosineAnnealingLR(classifier_optimizer,
                                            T_max = T_0,
                                            eta_min=eta_min,
                                            verbose=verbose)


    lr_schedulers = {"trunk_scheduler_by_epoch": trunk_lr,
                    "classifier_scheduler_by_epoch": classifier_lr}
    '''
    if args.lr_scheduler:
        verbose = True
        step_size = 90
        gamma = 0.6
        trunk_lr = torch.optim.lr_scheduler.StepLR(
            trunk_optimizer, step_size=step_size, gamma=gamma, last_epoch=-1, verbose=verbose)
        lr_schedulers = {"trunk_scheduler_by_epoch": trunk_lr}
        print('lr_scheduleStepLR step size', step_size)
        print('lr_scheduleStepLR gamm', gamma)
    else:
        lr_schedulers = None
        print('No lr schedule')

    print('model folder', model_folder)
    print('lr constant', args.lr_const)
    print('batch size', args.batch_size)
    print('len of train dataset', train_dataset.__len__())
    print('len of val dataset', val_dataset.__len__())
    print('number of epochs', args.epochs)
    print('miner margin', args.miner_margin)
    print('triplet_margin', args.triplet_margin)
    print('bn mom', args.bn_mom)
    print('trunk', args.backbone)
    print('trunk', trunk)
    print('embedder', embedder)
    print('classifer', classifier)
    print('train plates', args.train_plates)
    print('selected plates', args.select_plates)
    print('medium range', args.choose_range)
    #print ('len of train dataset with len()', len(train_dataset))

    if args.mode in ['train', 'val']:
        trainer = TrainWithClassifierCropHCS(models=models,
                                             optimizers=optimizers,
                                             loss_funcs=loss_funcs,
                                             mining_funcs=mining_funcs,
                                             batch_size=args.batch_size,
                                             dataset=train_dataset,
                                             dataloader_num_workers=args.num_data_workers,
                                             #data_and_label_getter = my_data_and_label_getter_24h,
                                             data_and_label_getter=data_and_label_getter,
                                             loss_weights=loss_weights,
                                             sampler=sampler,
                                             lr_schedulers=lr_schedulers,
                                             end_of_iteration_hook=hooks.end_of_iteration_hook,
                                             end_of_epoch_hook=end_of_epoch_hook,
                                             freeze_trunk_batchnorm=True,
                                             #freeze_these = ['trunk'],
                                             label_hierarchy_level="all",  # change to all
                                             # collate_fn = collate_fn, # for None case in dataset class
                                             crop=args.crop,
                                             avg_embs=args.avg_embs,
                                             )

        trainer.train(num_epochs=args.epochs)

    elif args.mode == 'predict':
        predict(args, model)
    else:
        assert 0


if __name__ == '__main__':
    args = parse_args()
    # setup_logging(args)
    setup_determinism(args)
    main(args)
