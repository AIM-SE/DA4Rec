# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.sequential_recommender.sasrecp import SASRecP
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import random


# Original SASRec Implementation
class ELECRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset, item_embedding=None):
        super(ELECRec, self).__init__(config, dataset)

        # load parameters info
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
        self.generator = SASRecP(config, dataset, self.item_embedding)
        self.discriminator = SASRecP(config, dataset, self.item_embedding)

        self.dis_loss_weight = 0.6  # 0.6
        self.sample_ratio = 0.45  # 0.45
        self.prob_power = 1
        self.project_type = "affine"
        self.dis_opt_versioin = "full"

        # Share is full
        self.discriminator.item_embedding = self.generator.item_embedding
        self.discriminator.position_embedding = self.generator.position_embedding
        self.discriminator.trm_encoder = self.generator.trm_encoder

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        self.m = nn.Softmax(dim=1)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        return self.generator.forward(item_seq, item_seq_len)  # [B L H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        item_label = item_seq[:, 1:]
        # pad = torch.zeros((item_label.size(0), 1)).to(self.device)
        pad = pos_items.unsqueeze(-1)
        item_labeln = torch.cat((item_label, pad), dim=-1).long().to(self.device)
        # idx = (item_seq_len.unsqueeze(-1)-1).to(self.device)
        # rep = pos_items.unsqueeze(-1)
        # item_labeln.scatter_(-1, idx, rep)
        sequence_output = seq_output

        # self.loss_type = 'CE'
        seq_output = seq_output[:, -1, :].squeeze(1)
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)

        target_pos = item_labeln

        # ---------- discriminator task -----------#
        (
            sampled_neg_ids,
            pos_idx,
            neg_idx,
            mask_idx,
            istarget,
        ) = self.sample_from_generator(sequence_output, target_pos)
        disc_logits = self.discriminator.forward(sampled_neg_ids, item_seq_len)
        dis_loss = self.discriminator_cross_entropy(
            disc_logits, pos_idx, neg_idx, mask_idx, istarget
        )

        return loss, self.dis_loss_weight * dis_loss

    def _generate_sample(self, probability, pos_ids, neg_ids, neg_nums):
        neg_ids = neg_ids.expand(probability.shape[0], -1)
        # try:
        neg_idxs = torch.multinomial(probability, neg_nums).to(self.device)

        neg_ids = torch.gather(neg_ids, 1, neg_idxs)
        neg_ids = neg_ids.view(-1, self.max_seq_length)
        # replace the sampled positive ids with uniform sampled items
        return neg_ids

    def sample_from_generator(self, seq_out, pos_ids):
        seq_emb = seq_out.view(-1, self.hidden_size)
        istarget = (
            (pos_ids > 0).view(pos_ids.size(0) * self.max_seq_length).float()
        )  # [batch*seq_len]
        K = int(self.n_items * self.sample_ratio) - 1
        neg_ids = random.sample([i for i in range(1, self.n_items)], K)
        neg_ids = torch.tensor(neg_ids, dtype=torch.long).to(self.device)
        neg_emb = self.generator.item_embedding(neg_ids)
        full_probability = torch.matmul(seq_emb, neg_emb.transpose(0, 1))
        full_probability = self.m(full_probability) ** self.prob_power
        sampled_neg_ids = self._generate_sample(full_probability, pos_ids, neg_ids, 1)

        # replace certain percentage of items as absolute positive items
        replace_idx = torch.rand(size=(pos_ids.size(0), pos_ids.size(1))) < (
            1 - self.sample_ratio
        )
        sampled_neg_ids[replace_idx] = pos_ids[replace_idx]
        mask_idx = torch.logical_not(replace_idx).float()
        pos_idx = (
            (pos_ids == sampled_neg_ids)
            .view(pos_ids.size(0) * self.generator.max_seq_length)
            .float()
        )
        neg_idx = (
            (pos_ids != sampled_neg_ids)
            .view(pos_ids.size(0) * self.generator.max_seq_length)
            .float()
        )
        return sampled_neg_ids, pos_idx, neg_idx, mask_idx, istarget

    def discriminator_cross_entropy(
        self, seq_out, pos_idx, neg_idx, mask_idx, istarget
    ):
        seq_emb = seq_out.view(-1, self.hidden_size)
        # sum over feature di  m
        if self.project_type == "sum":
            neg_logits = torch.sum((seq_emb) / self.temperature, -1)
        elif self.project_type == "affine":
            neg_logits = torch.squeeze(self.discriminator.dis_projection(seq_emb))

        prob_score = torch.sigmoid(neg_logits) + 1e-24
        if self.dis_opt_versioin == "mask_only":
            total_pos_loss = torch.log(prob_score) * istarget * pos_idx * mask_idx
            total_neg_loss = torch.log(1 - prob_score) * istarget * neg_idx * mask_idx
        else:
            total_pos_loss = torch.log(prob_score) * istarget * pos_idx
            total_neg_loss = torch.log(1 - prob_score) * istarget * neg_idx

        loss = torch.sum(-total_pos_loss - total_neg_loss) / (torch.sum(istarget))
        return loss

    def cross_entropy(self, seq_out, pos_ids):
        # [batch seq_len hidden_size]
        seq_emb = seq_out.view(-1, self.hidden_size)  # [batch*seq_len hidden_size]
        # istarget = (pos_ids > 0).view(pos_ids.size(0) * self.max_seq_length).float()  # [batch*seq_len]

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_emb, test_item_emb.transpose(0, 1))
        pos_ids_l = torch.squeeze(pos_ids.view(-1))
        loss = self.loss_fct(logits, pos_ids_l)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.discriminator.forward(item_seq, item_seq_len)
        # seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        seq_output = seq_output[:, -1, :].squeeze(1)

        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.discriminator.forward(item_seq, item_seq_len)
        # seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        seq_output = seq_output[:, -1, :].squeeze(1)

        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
