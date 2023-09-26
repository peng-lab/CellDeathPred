import torch
from pytorch_metric_learning.miners import BaseTupleMiner
from pytorch_metric_learning.utils import common_functions as c_f
#from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import numpy as np

def subset_tensor_ind(t, t1,t2, n1,n2):
    a_p = torch.eq(t1, n1).nonzero() 
    a_d = torch.eq(t2, n2).nonzero()
    subset_ind = torch.tensor(np.intersect1d(a_p, a_d))
    return subset_ind

def my_get_all_triplets_indices(labels):
    plates, drug_types = torch.tensor(list(zip(*labels)))
    #print ('plates', plates)
    #print ('drug_types', drug_types)
    train_plates = torch.unique(plates,sorted=True)
    #print (train_plates)
    anchor_set_labels = [] 
    pos_set_labels = [] 
    neg_set_labels = [] 
    for drug in [0,1,2]:
        drug_unique = [0,1,2]
        anchor_set_labels.append(subset_tensor_ind(labels, plates, drug_types, train_plates[0], drug))
        pos_set_labels.append(subset_tensor_ind(labels, plates, drug_types, train_plates[1], drug))
        drug_unique.remove(drug)
        for neg_drug in drug_unique:
            neg_set_labels.append(subset_tensor_ind(labels, plates, drug_types, train_plates[0], neg_drug))
    return torch.cat(anchor_set_labels).repeat(2) , torch.cat(pos_set_labels).repeat(2), torch.cat(neg_set_labels)
    
class TripletMinerHCS(BaseTupleMiner):
    """
    Returns triplets that violate the margin
    Args:
        margin
        type_of_triplets: options are "all", "hard", or "semihard".
                "all" means all triplets that violate the margin
                "hard" is a subset of "all", but the negative is closer to the anchor than the positive
                "semihard" is a subset of "all", but the negative is further from the anchor than the positive
            "easy" is all triplets that are not in "all"
    """

    def __init__(self, margin=0.1, type_of_triplets="all", **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.type_of_triplets = type_of_triplets
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)
        self.add_to_recordable_attributes(
            list_of_names=["avg_triplet_margin", "pos_pair_dist", "neg_pair_dist"],
            is_stat=True,
        )
    
    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
        Does any necessary preprocessing, then does mining, and then checks the
        shape of the mining output before returning it
        """
        self.reset_stats()
        with torch.no_grad():
            #c_f.check_shapes(embeddings, labels)
            labels = c_f.to_device(labels, embeddings)
            ref_emb, ref_labels = self.set_ref_emb(
                embeddings, labels, ref_emb, ref_labels
            )
            mining_output = self.mine(embeddings, labels, ref_emb, ref_labels)
        self.output_assertion(mining_output)
        return mining_output
    
    def set_ref_emb(self, embeddings, labels, ref_emb, ref_labels):
        if ref_emb is not None:
            ref_labels = c_f.to_device(ref_labels, ref_emb)
        else:
            ref_emb, ref_labels = embeddings, labels
        #c_f.check_shapes(ref_emb, ref_labels)
        return ref_emb, ref_labels
        
    def mine(self, embeddings, labels, ref_emb, ref_labels):
        #anchor_idx, positive_idx, negative_idx = lmu.get_all_triplets_indices(labels, ref_labels)
        anchor_idx, positive_idx, negative_idx = my_get_all_triplets_indices(labels)
        #print (len(anchor_idx))
        #print (len(positive_idx))
        #print (len(negative_idx))
        mat = self.distance(embeddings, ref_emb)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]
        triplet_margin = (
            ap_dist - an_dist if self.distance.is_inverted else an_dist - ap_dist
        )
        # maybe add "hard" option?
        if self.type_of_triplets == "easy":
            threshold_condition = triplet_margin > self.margin
        else:
            threshold_condition = triplet_margin <= self.margin
            if self.type_of_triplets == "hard":
                threshold_condition &= triplet_margin <= 0
            elif self.type_of_triplets == "semihard":
                threshold_condition &= triplet_margin > 0
        
        return (
            anchor_idx[threshold_condition],
            positive_idx[threshold_condition],
            negative_idx[threshold_condition],
        )
    
    def set_stats(self, ap_dist, an_dist, triplet_margin):
        if self.collect_stats:
            with torch.no_grad():
                self.pos_pair_dist = torch.mean(ap_dist).item()
                self.neg_pair_dist = torch.mean(an_dist).item()
                self.avg_triplet_margin = torch.mean(triplet_margin).item()
                
   
    
