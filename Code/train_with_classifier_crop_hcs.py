import math
import random
import os

import tqdm
import numpy as np
import torch
from torch import nn

from pytorch_metric_learning.utils import common_functions as c_f
#from ..utils import common_functions as c_f

from pytorch_metric_learning.trainers import TrainWithClassifier

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



class TrainWithClassifierCropHCS(TrainWithClassifier):
    def __init__(self, crop, avg_embs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.embedding_sizes = embedding_sizes 
        self.crop = crop
        self.avg_embs = avg_embs
    
    
    @torch.no_grad()
    def transform_input(self, X, Y, mixup=0, cutmix=1):
        """Apply mixup, cutmix, and label-smoothing"""

        Y = smooth_label(Y)
        
        if mixup != 0 or cutmix != 0:
            perm = torch.randperm(self.batch_size).cuda()

        if mixup != 0:
            coeffs = torch.tensor(np.random.beta(
                mixup, mixup, self.batch_size), dtype=torch.float32).cuda()
            X = coeffs.view(-1, 1, 1, 1) * X + \
                (1 - coeffs.view(-1, 1, 1, 1)) * X[perm, ]
            #S = coeffs.view(-1, 1) * S + (1 - coeffs.view(-1, 1)) * S[perm, ]
            Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm, ]

        if cutmix != 0:
            img_height, img_width = X.size()[2:]
            lambd = np.random.beta(cutmix, cutmix)
            column = np.random.uniform(0, img_width)
            row = np.random.uniform(0, img_height)
            height = (1 - lambd) ** 0.5 * img_height
            width = (1 - lambd) ** 0.5 * img_width
            r1 = round(max(0, row - height / 2))
            r2 = round(min(img_height, row + height / 2))
            c1 = round(max(0, column - width / 2))
            c2 = round(min(img_width, column + width / 2))
            if r1 < r2 and c1 < c2:
                X[:, :, r1:r2, c1:c2] = X[perm, :, r1:r2, c1:c2]

                lambd = 1 - (r2 - r1) * (c2 - c1) / (img_height * img_width)
                #S = S * lambd + S[perm] * (1 - lambd)
                Y = Y * lambd + Y[perm] * (1 - lambd)
        
        return X, Y
    
    

    @torch.no_grad()
    def infer(self, loader):
        """Infer and return prediction in dictionary formatted {sample_id: logits}"""

        if not len(loader):
            return {}
        res = {}
        
        for m in self.models.values():
            m.eval()
        
        for i, (X, _, _, I, *_) in enumerate(loader):
            X = X.to(self.data_device)
            #Xs = dataset.tta(args, X) if args.tta else [X]
            Xs = tta(X)
            ys = [self.models['classifier'].module.forward(self.models['trunk'].module.forward(X)) for X in Xs]
            #ys = [model.module.forward(X) for X in Xs]
            y = torch.stack(ys).mean(0).cpu()

            for j in range(len(I)):
                assert I[j].item() not in res
                res[I[j].item()] = y[j].numpy()

        return res    
    
    def score(self, dataset):
        """Return accuracy of the model on validation set"""

        print('Starting validation')
        
        loader = DataLoader(dataset, batch_size = self.batch_size, shuffle=False, num_workers=self.dataloader_num_workers)

        res = self.infer(loader)
        
        n = 0
        s = 0
        for i, v in res.items():
            d = loader.dataset.data[i]
            r = v[:loader.dataset.treatment_classes].argmax() == d[-1]

            n += 1
            s += r

        acc = s / n if n != 0 else 0
        print('Eval: acc: {}'.format(acc))

        return acc
    
    
    
    def train(self, start_epoch=1, num_epochs=1, checkpoint_dir = None):
        #ipdb.set_trace()

        self.initialize_dataloader()
        
        # Creates a GradScaler once at the beginning of training.
        #self.scaler = torch.cuda.amp.GradScaler()
        best_acc = float('-inf')
        
        for self.epoch in range(start_epoch, num_epochs + 1):
            #ipdb.set_trace()
            '''
            with torch.no_grad():
                avg_norm = np.mean([v.norm().item() for v in self.models['trunk'].parameters()])
            print (f'epoch{self.epoch}', f'  avg_norm{    avg_norm}')
            '''
            self.acc = 0
            #self.loss = 0
            self.cum_count = 0

            print (f'epoch{self.epoch}')
            self.set_to_train()
            c_f.LOGGER.info("TRAINING EPOCH %d" % self.epoch)
            pbar = tqdm.tqdm(range(self.iterations_per_epoch))

            #self.zero_grad()  # here?

            for self.iteration in pbar:
                #ipdb.set_trace()
                self.forward_and_backward()
                self.end_of_iteration_hook(self)
                pbar.set_description("total_loss=%.5f" %
                                     self.losses["total_loss"])
                #print ('loss', self.losses["total_loss"])
                self.step_lr_schedulers(end_of_epoch=False)
            self.step_lr_schedulers(end_of_epoch=True)
            
            #tune.report(loss = self.losses["total_loss"].item(), epoch=self.epoch)
            
            self.zero_losses()
            '''
            models_names = list(self.models)
            i = 0
            for m in self.models.values():
                torch.save(m.state_dict(), str("/storage/groups/peng/projects/aidin_Kenjis_project/src/saved_models/circle_loss_try1" + '.{}'.format(models_names[i]) + '.{}'.format(self.epoch)))
                i += 1
            '''
            '''
            acc = self.score(dataset = self.dataset)
            if acc >= best_acc:
                best_acc = acc
                models_names = list(self.models)
                i = 0
                for m in self.models.values():
                    torch.save(m.state_dict(), str("/storage/groups/peng/projects/aidin_Kenjis_project/hcs_atp/saved_models/" + 'wometric_stain1.{}'.format(models_names[i]) + '.{}'.format(self.epoch)))
                    i += 1
                
                print('Saving best model with score {}'.format(best_acc))
            '''
            
            '''
            with tune.checkpoint_dir(self.epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                #models_names = list(trainer.models)
                i = 0
                for m in self.models.values():
                    torch.save(m.state_dict(), path)
                    i += 1  
            '''
            if self.end_of_epoch_hook(self) is False:
                break

    
    def forward_and_backward(self):
        '''
        lr = get_learning_rate(self.epoch + self.iteration / len(self.dataloader))
        for optimizer in self.optimizers.values():
            for g in optimizer.param_groups:
                g['lr'] = lr
        '''
        self.zero_losses() # problem is here?
        self.zero_grad() # turn off when using mixed precision
        self.update_loss_weights()
        self.calculate_loss(self.get_batch())
        
        #print("unweighted total_loss: ", self.losses["total_loss"],
         #     "unweighted metric_loss: ", self.losses["metric_loss"],
          #    "unweighted classifier_loss: ", self.losses["classifier_loss"])
        
        self.loss_tracker.update(self.loss_weights) # the calc. of total loss is done here(with the weights)

        #self.scaler.scale(self.losses["metric_loss"]).backward(
            #retain_graph=True)
        #self.scaler.scale(self.losses["classifier_loss"]).backward(
            #retain_graph=True)

        #self.scaler.scale(self.losses["total_loss"]).backward()
        self.backward()
        
        #for m in self.models.values(): 
        #    plot_grad_flow(m.named_parameters()) #check of gradient flow 


        self.clip_gradients()
        self.step_optimizers()

        '''
        if (self.iteration + 1) % 2 == 0:  # number of iterations for gradient accumulation
            # self.step_optimizers(scaler)
            for v in self.optimizers.values():
                self.scaler.step(v)

            self.scaler.update()

            self.zero_grad()
        '''
        '''

        self.acc += acc
        self.cum_count += 1
        
        if (self.iteration + 1) % 50 == 0:
            print("total_loss: ", self.losses["total_loss"],
                "metric_loss: ", self.losses["metric_loss"],
                "classifier_loss: ", self.losses["classifier_loss"])
            print("acc: ", self.acc / self.cum_count)
            #print("learning rate: ", lr)
            
            self.acc = 0
            self.cum_count = 0
                    '''

        '''
        self.acc += acc
        self.loss += self.losses["total_loss"].item() # extracts the loss’s value as a Python float
        self.cum_count += 1
        
        if (self.iteration + 1) % 50 == 0:
            #logging.info('Epoch: {:3d} Iter: {:4d} lr: {:.9f}   loss: {:.6f}   acc: {:.6f}'.format(
                    #self.epoch, self.iteration + 1,  self.optimizers['trunk_optimizer'].param_groups[0]['lr'],
                    #self.loss / self.cum_count, self.acc / self.cum_count))
            
            
            print ("total_loss.item() / cum_count", self.loss / self.cum_count)
            print("acc: ", self.acc / self.cum_count)
            
            #print("learning rate: ", lr)
            
            #ipdb.set_trace()
            
            self.acc = 0
            self.loss = 0
            self.cum_count = 0
    '''
    
    
    # change this method for correctly getting batches from self.dataloader
    def get_batch(self):
        #with torch.cuda.amp.autocast():
        #ipdb.set_trace()
        self.dataloader_iter, curr_batch = c_f.try_next_on_generator(
            self.dataloader_iter, self.dataloader
        )
        #print ('current batch', curr_batch)
        data, labels = self.data_and_label_getter(curr_batch)
        
        #data = data.to(self.data_device)
        #labels = labels.to(self.data_device)
        #data = c_f.to_device(data, device=self.data_device, dtype=self.dtype)
        #labels = c_f.to_device(labels, device=self.data_device, dtype=self.dtype)
        
        #data, labels_transformed = self.transform_input(data, labels) # labels_transformed need for acc & classifier_loss
        
        #print (' data:', data.shape)
        #print ('plate, drug_type:', labels)
        labels = c_f.process_label(
            labels, self.label_hierarchy_level, self.label_mapper
        ) # after this labels are drug_type
        #print ('after labels:', labels.shape)
        return self.maybe_do_batch_mining(data, labels)
        
        '''
        # curr_batch[0].shape = torch.Size([24, 6, 512, 512]
        X, S, labels = self.data_and_label_getter(curr_batch) 

        X = X.to(self.data_device)
        S = S.to(self.data_device)
        labels = labels.to(self.data_device)

            #print ("before trasform: ", labels)
            #print ("shape of labels before transform: ", labels.shape)

        X, S, labels_transformed = self.transform_input(X, S, labels)

            #print ("after trasform: ", labels)
        shape = labels_transformed.shape # [24, 1139]
            #print ("shape of labels after transform: ", shape)
            # labels.resize_((shape[0]))  # here is the problem?
            # print(labels)

            #labels_wo_process_label = labels

        #for the metric loss calculation I've created a separate variable for the processed labels
        # labels_processed = labels
        labels_processed = c_f.process_label(
        labels, self.label_hierarchy_level, self.label_mapper
        ) # not in cuda device

        #print("labels after c_f.process_label: ", labels)
        #print ("shape of labels after c_f.process_label:: ", labels.shape)

        return self.maybe_do_batch_mining((X, S, shape, labels_processed), labels_transformed)
        '''
    
    
    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        
        #result = model(data.view(-1, c, h, w)) # fuse batch size and ncrops
        #result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
        #with torch.cuda.amp.autocast():
        
        #print ('data with crops shape:', data.shape)
        #print ('data with crops:', data)
        #print ('labels:', labels)
        #print ('labels shape:', labels.shape)
        # avg embs over crops! or maybe need to avg logits?
        if self.crop:
            bs, ncrops, c, h, w = data.size()
            if self.avg_embs:
                embeddings = self.compute_embeddings(data.view(-1, c, h, w)).view(bs, ncrops, -1).mean(1) # dtype=torch.float16
            else:
                #print ("berofe labels: ", labels.shape)
                embeddings = self.compute_embeddings(data.view(-1, c, h, w))
                #labels = labels.repeat(5) # 5=#crops
                labels = labels.repeat_interleave(5, dim=0)
                #print ("embeddings: ", embeddings.shape)
                #print ("labels: ", labels.shape)
        else:
            embeddings = self.compute_embeddings(data) 
                
                
        logits = self.maybe_get_logits(embeddings)
        
        #indices_tuple = self.maybe_mine_embeddings(embeddings, labels, ref_emb, ref_labels) # check if triplets are formed correctly!
        
        
        plates, drug_types = torch.tensor(list(zip(*labels)))
        
        #print ('plates:', plates)
        #print ('drug_types:', drug_types)
        indices_tuple = self.maybe_mine_embeddings(embeddings, labels) # check if triplets are formed correctly!
        
        #print ('indices_tuple:', indices_tuple)
        #print ('indices_tuple shape of element:', indices_tuple[0].shape)
                
        # since passing non empty indices_tuple labels = drug_types does not matter       
        self.losses["metric_loss"] = self.maybe_get_metric_loss(
            embeddings, drug_types, indices_tuple) 
        self.losses["classifier_loss"] = self.maybe_get_classifier_loss(logits, drug_types) 
        
    def maybe_get_classifier_loss(self, logits, labels):
        if logits is not None:
            return self.loss_funcs["classifier_loss"](
                logits, c_f.to_device(labels, logits)
            )
        return 0

    def maybe_get_logits(self, embeddings):
        if (
            self.models.get("classifier", None)
            and self.loss_weights.get("classifier_loss", 0) > 0
        ):
            return self.models["classifier"](embeddings)
        return None

    def modify_schema(self):
        self.schema["models"].keys += ["classifier"]
        self.schema["loss_funcs"].keys += ["classifier_loss"]
    '''
    def compute_embeddings(self, data):
        # data <-> (X, S)
        trunk_output = self.get_trunk_output(data[0]) # check if data[0] is actually X
        #S = c_f.to_device(data[1], device=self.data_device, dtype=self.dtype) # to cuda; same as data[0]
        embeddings = self.get_final_embeddings(trunk_output, data[1])
        return embeddings
    
    def get_final_embeddings(self, base_output, s):
        return self.models["embedder"](base_output, s)

'''
    
   
