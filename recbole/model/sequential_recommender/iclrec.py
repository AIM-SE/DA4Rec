# -*- coding: utf-8 -*-
# @Time    : 2022/08/15 14:34
# @Author  : Qichen YE
# @Email   : yeeeqichen@pku.edu.cn

r"""
ICLRec: Intent Contrastive Learning for Sequential Recommendation
paper link: https://arxiv.org/pdf/2202.02519.pdf
official implementation: https://github.com/salesforce/ICLRec
################################################
"""


import math
import os
import pickle
from tqdm import tqdm
import random
import copy

import torch
import torch.nn as nn
import gensim
import faiss

import time

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder


def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)


class Random(object):
    """Randomly pick one data augmentation type every time call"""

    def __init__(self, tao=0.2, gamma=0.7, beta=0.2):
        self.data_augmentation_methods = [
            Crop(tao=tao),
            Mask(gamma=gamma),
            Reorder(beta=beta),
        ]
        # print("[ICLRec] Total augmentation numbers: ", len(self.data_augmentation_methods))

    def __call__(self, sequence):
        # randint generate int x in range: a <= x <= b
        augment_method_idx = random.randint(0, len(self.data_augmentation_methods) - 1)
        augment_method = self.data_augmentation_methods[augment_method_idx]
        # print(augment_method.__class__.__name__) # debug usage
        return augment_method(sequence)


class Crop(object):
    """Randomly crop a subseq from the original sequence"""

    def __init__(self, tao=0.2):
        self.tao = tao

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        if len(copied_sequence) == 0:
            return copied_sequence
        sub_seq_length = int(self.tao * len(copied_sequence))
        # randint generate int x in range: a <= x <= b
        # print(copied_sequence)
        start_index = random.randint(
            0, max(len(copied_sequence) - sub_seq_length - 1, 0)
        )
        if sub_seq_length < 1:
            return [copied_sequence[start_index]]
        else:
            cropped_seq = copied_sequence[start_index : start_index + sub_seq_length]
            return cropped_seq


class Mask(object):
    """Randomly mask k items given a sequence"""

    def __init__(self, gamma=0.7):
        self.gamma = gamma

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        mask_nums = int(self.gamma * len(copied_sequence))
        mask = [0 for i in range(mask_nums)]
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
        for idx, mask_value in zip(mask_idx, mask):
            copied_sequence[idx] = mask_value
        return copied_sequence


class Reorder(object):
    """Randomly shuffle a continuous sub-sequence"""

    def __init__(self, beta=0.2):
        self.beta = beta

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        if len(copied_sequence) == 0:
            return copied_sequence
        sub_seq_length = int(self.beta * len(copied_sequence))
        start_index = random.randint(
            0, max(len(copied_sequence) - sub_seq_length - 1, 0)
        )
        sub_seq = copied_sequence[start_index : start_index + sub_seq_length]
        random.shuffle(sub_seq)
        reordered_seq = (
            copied_sequence[:start_index]
            + sub_seq
            + copied_sequence[start_index + sub_seq_length :]
        )
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq


class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []

    def __init_cluster(
        self,
        hidden_size,
        verbose=False,
        niter=20,
        nredo=5,
        max_points_per_centroid=4096,
        min_points_per_centroid=0,
    ):
        # print("[ICLRec] cluster train iterations:", niter)
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        #
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)

        # index = faiss.IndexFlatL2(hidden_size)
        return clus, index

    def train(self, x):
        # train to get centroids
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)
        # get cluster centroids
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(
            self.num_cluster, self.hidden_size
        )
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

    def query(self, x):
        # self.index.add(x)
        D, I = self.index.search(
            x, 1
        )  # for each sample, find cluster distance and assignments
        seq2cluster = [int(n[0]) for n in I]
        # print("cluster number:", self.num_cluster,"cluster in batch:", len(set(seq2cluster)))
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, self.centroids[seq2cluster]


class KMeans_Pytorch(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 10
        self.first_batch = True
        self.hidden_size = hidden_size
        self.gpu_id = gpu_id
        self.device = device
        # print(self.device, "-----")

    def run_kmeans(self, x, Niter=20, tqdm_flag=False):
        if x.shape[0] >= self.num_cluster:
            seq2cluster, centroids = kmeans(
                X=x,
                num_clusters=self.num_cluster,
                distance="euclidean",
                device=self.device,
                tqdm_flag=False,
            )
            seq2cluster = seq2cluster.to(self.device)
            centroids = centroids.to(self.device)
        # last batch where
        else:
            seq2cluster, centroids = kmeans(
                X=x,
                num_clusters=x.shape[0] - 1,
                distance="euclidean",
                device=self.device,
                tqdm_flag=False,
            )
            seq2cluster = seq2cluster.to(self.device)
            centroids = centroids.to(self.device)
        return seq2cluster, centroids


class NCELoss(nn.Module):
    """
    Eq. (12): L_{NCE}
    """

    def __init__(self, temperature, device):
        super(NCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)

    # #modified based on impl: https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
    def forward(self, batch_sample_one, batch_sample_two, intent_ids=None):
        # sim11 = self.cossim(batch_sample_one.unsqueeze(-2), batch_sample_one.unsqueeze(-3)) / self.temperature
        # sim22 = self.cossim(batch_sample_two.unsqueeze(-2), batch_sample_two.unsqueeze(-3)) / self.temperature
        # sim12 = self.cossim(batch_sample_one.unsqueeze(-2), batch_sample_two.unsqueeze(-3)) / self.temperature
        sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        # avoid contrast against positive intents
        if intent_ids is not None:
            intent_ids = intent_ids.contiguous().view(-1, 1)
            mask_11_22 = torch.eq(intent_ids, intent_ids.T).long().to(self.device)
            sim11[mask_11_22 == 1] = float("-inf")
            sim22[mask_11_22 == 1] = float("-inf")
            eye_metrix = torch.eye(d, dtype=torch.long).to(self.device)
            mask_11_22[eye_metrix == 1] = 0
            sim12[mask_11_22 == 1] = float("-inf")
        else:
            mask = torch.eye(d, dtype=torch.long).to(self.device)
            sim11[mask == 1] = float("-inf")
            sim22[mask == 1] = float("-inf")
            # sim22 = sim22.masked_fill_(mask, -np.inf)
            # sim11[..., range(d), range(d)] = float('-inf')
            # sim22[..., range(d), range(d)] = float('-inf')

        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss


class PCLoss(nn.Module):
    """Reference: https://github.com/salesforce/PCL/blob/018a929c53fcb93fd07041b1725185e1237d2c0e/pcl/builder.py#L168"""

    def __init__(self, temperature, device, contrast_mode="all"):
        super(PCLoss, self).__init__()
        self.contrast_mode = contrast_mode
        self.criterion = NCELoss(temperature, device)

    def forward(self, batch_sample_one, batch_sample_two, intents, intent_ids):
        """
        features:
        intents: num_clusters x batch_size x hidden_dims
        """
        # instance contrast with prototypes
        mean_pcl_loss = 0
        # do de-noise
        if intent_ids is not None:
            for intent, intent_id in zip(intents, intent_ids):
                pos_one_compare_loss = self.criterion(
                    batch_sample_one, intent, intent_id
                )
                pos_two_compare_loss = self.criterion(
                    batch_sample_two, intent, intent_id
                )
                mean_pcl_loss += pos_one_compare_loss
                mean_pcl_loss += pos_two_compare_loss
            mean_pcl_loss /= 2 * len(intents)
        # don't do de-noise
        else:
            for intent in intents:
                pos_one_compare_loss = self.criterion(
                    batch_sample_one, intent, intent_ids=None
                )
                pos_two_compare_loss = self.criterion(
                    batch_sample_two, intent, intent_ids=None
                )
                mean_pcl_loss += pos_one_compare_loss
                mean_pcl_loss += pos_two_compare_loss
            mean_pcl_loss /= 2 * len(intents)
        return mean_pcl_loss


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class ICLRec(SequentialRecommender):
    def __init__(self, args, dataset):
        super(ICLRec, self).__init__(args, dataset)
        config = args
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.max_len = args["MAX_ITEM_LIST_LENGTH"]
        self.n_views = args["n_views"]
        self.de_noise = args["de_noise"]
        self.augmentations = {
            "crop": Crop(tao=args["tao"]),
            "mask": Mask(gamma=args["gamma"]),
            "reorder": Reorder(beta=args["beta"]),
            "random": Random(tao=args["tao"], gamma=args["gamma"], beta=args["beta"]),
        }
        self.augment_type = args["augment_type"]
        self.base_transform = self.augmentations[self.augment_type]

        self.num_intent_clusters = [
            int(i) for i in self.args.num_intent_clusters.split(",")
        ]
        self.clusters = []
        for num_intent_cluster in self.num_intent_clusters:
            # initialize Kmeans
            if self.args.seq_representation_type == "mean":
                cluster = KMeans(
                    num_cluster=num_intent_cluster,
                    seed=775,
                    hidden_size=self.args.hidden_size,
                    gpu_id=self.args.gpu_id,
                    device=self.device,
                )
                self.clusters.append(cluster)
            else:
                cluster = KMeans(
                    num_cluster=num_intent_cluster,
                    seed=775,
                    hidden_size=self.args.hidden_size * self.max_seq_length,
                    gpu_id=self.args.gpu_id,
                    device=self.device,
                )
                self.clusters.append(cluster)
        self.loss_fct = nn.CrossEntropyLoss()
        self.cf_criterion = NCELoss(self.args.temperature, device=self.device)
        self.pcl_criterion = PCLoss(self.args.temperature, self.device)
        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # model same as SASRec
    def forward(self, item_seq):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # extended_attention_mask = self.get_attention_mask(item_seq)

        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2
        )  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.use_gpu:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        # output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B , seq, H]

    def _one_pair_data_augmentation(self, input_ids):
        """
        provides two positive samples given one sequence
        """
        augmented_seqs = []
        for i in range(2):
            augmented_input_ids = self.base_transform(input_ids)
            pad_len = self.max_len - len(augmented_input_ids)
            augmented_input_ids = [0] * pad_len + augmented_input_ids

            augmented_input_ids = augmented_input_ids[-self.max_len :]

            assert len(augmented_input_ids) == self.max_len

            cur_tensors = torch.tensor(augmented_input_ids, dtype=torch.long)
            augmented_seqs.append(cur_tensors)
        return augmented_seqs

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in copied_input_ids:
            target_neg.append(neg_sample(seq_set, self.n_items))

        pad_len = self.max_len - len(copied_input_ids)
        # print(pad_len)
        copied_input_ids = [0] * pad_len + copied_input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg
        # print(len(copied_input_ids))
        copied_input_ids = copied_input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]

        assert len(copied_input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(copied_input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

        return cur_rec_tensors

    def _generate_inputs(self, interactions):
        batch_rec_tensors = []
        batch_cf_tensors = []
        batch_seq_class_label = []
        total_augmentaion_pairs = nCr(self.n_views, 2)
        for i, items in enumerate(interactions[self.ITEM_SEQ].cpu().numpy()):
            # print(items > 0)
            length = interactions[self.ITEM_SEQ_LEN][i]
            items = list(items[:length])
            # print(items)
            input_ids = items[:-1]
            target_pos = items[1:]
            seq_label_signal = interactions[self.ITEM_ID][i]
            answer = [0]
            cur_rec_tensors = self._data_sample_rec_task(
                i, items, input_ids, target_pos, answer
            )
            cf_tensors_list = []
            for i in range(total_augmentaion_pairs):
                cf_tensors_list.append(self._one_pair_data_augmentation(input_ids))
            # seq_class_label = torch.tensor(seq_label_signal, dtype=torch.long)
            seq_class_label = seq_label_signal.clone().detach()
            batch_seq_class_label.append(seq_class_label)
            batch_rec_tensors.append(cur_rec_tensors)
            batch_cf_tensors.append(cf_tensors_list)
        batch_seq_class_label = torch.stack(batch_seq_class_label, dim=0)
        temp = []
        for i in range(len(batch_rec_tensors[0])):
            temp.append(torch.stack([_[i] for _ in batch_rec_tensors], dim=0))
        batch_rec_tensors = temp
        temp = []
        for i in range(total_augmentaion_pairs):
            ttemp = []
            for j in range(len(batch_cf_tensors[0][i])):
                ttemp.append(torch.stack([_[i][j] for _ in batch_cf_tensors], dim=0))
            temp.append(ttemp)
        batch_cf_tensors = temp

        return batch_rec_tensors, batch_cf_tensors, batch_seq_class_label

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.item_embeddings(pos_ids)
        neg_emb = self.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (
            (pos_ids > 0).view(pos_ids.size(0) * self.max_seq_length).float()
        )  # [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def _instance_cl_one_pair_contrastive_learning(self, inputs, intent_ids=None):
        """
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        """
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.forward(cl_batch)
        # cf_sequence_output = cf_sequence_output[:, -1, :]
        if self.args.seq_representation_instancecl_type == "mean":
            cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0] // 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        if self.de_noise:
            cl_loss = self.cf_criterion(
                cl_output_slice[0], cl_output_slice[1], intent_ids=intent_ids
            )
        else:
            cl_loss = self.cf_criterion(
                cl_output_slice[0], cl_output_slice[1], intent_ids=None
            )
        return cl_loss

    def _pcl_one_pair_contrastive_learning(self, inputs, intents, intent_ids):
        """
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        intents: [num_clusters batch_size hidden_dims]
        """
        n_views, (bsz, seq_len) = len(inputs), inputs[0].shape
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.forward(cl_batch)
        if self.args.seq_representation_type == "mean":
            cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        cl_output_slice = torch.split(cl_sequence_flatten, bsz)
        if self.args.de_noise:
            cl_loss = self.pcl_criterion(
                cl_output_slice[0],
                cl_output_slice[1],
                intents=intents,
                intent_ids=intent_ids,
            )
        else:
            cl_loss = self.pcl_criterion(
                cl_output_slice[0], cl_output_slice[1], intents=intents, intent_ids=None
            )
        return cl_loss

    def get_seq_output(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        # item_seq_len = interaction[self.ITEM_SEQ_LEN]
        sequence_output = self.forward(item_seq)
        # print(sequence_output.shape)
        return sequence_output

    def calculate_loss(self, interaction, warm_up=False):
        rec_batch, cl_batches, seq_class_label_batches = self._generate_inputs(
            interaction
        )
        rec_batch = tuple(t.to(self.device) for t in rec_batch)
        _, input_ids, target_pos, target_neg, _ = rec_batch
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        sequence_output = self.forward(item_seq)
        seq_output = self.gather_indexes(sequence_output, item_seq_len - 1)
        # ---------- recommendation task ---------------#
        # sequence_output = self.forward(input_ids)
        # rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        rec_loss = self.loss_fct(logits, pos_items)
        # print(rec_loss)
        # exit('debug')
        # ---------- contrastive learning task -------------#
        cl_losses = []
        for cl_batch in cl_batches:
            assert self.args.contrast_type == "Hybrid"
            if warm_up:
                cl_loss1 = self._instance_cl_one_pair_contrastive_learning(
                    cl_batch, intent_ids=seq_class_label_batches
                )
                cl_losses.append(self.args.cf_weight * cl_loss1)
            else:
                cl_loss1 = self._instance_cl_one_pair_contrastive_learning(
                    cl_batch, intent_ids=seq_class_label_batches
                )
                cl_losses.append(self.args.cf_weight * cl_loss1)
                if self.args.seq_representation_type == "mean":
                    sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
                sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                sequence_output = sequence_output.detach().cpu().numpy()
                # query on multiple clusters
                for cluster in self.clusters:
                    seq2intents = []
                    intent_ids = []
                    intent_id, seq2intent = cluster.query(sequence_output)
                    seq2intents.append(seq2intent)
                    intent_ids.append(intent_id)
                cl_loss3 = self._pcl_one_pair_contrastive_learning(
                    cl_batch, intents=seq2intents, intent_ids=intent_ids
                )
                cl_losses.append(self.args.intent_cf_weight * cl_loss3)
        joint_loss = self.args.rec_weight * rec_loss
        for cl_loss in cl_losses:
            joint_loss += cl_loss
        return joint_loss

    def predict(self, interaction):
        raise NotImplementedError

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.get_seq_output(interaction)
        # print(seq_output.shape)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        # test_items_emb = self.item_embeddings.weight[:self.n_items - 1]  # delete masked token
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        # print(scores.shape)
        # exit('debug')
        return scores
