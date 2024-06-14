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
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


# Original SASRec Implementation
class SASRecP(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset, item_embedding=None):
        super(SASRecP, self).__init__(config, dataset)

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

        self.ignore_pad = config["ignore_pad"]

        # define layers and loss
        if item_embedding:
            self.item_embedding = item_embedding
        else:
            self.item_embedding = nn.Embedding(
                self.n_items, self.hidden_size, padding_idx=0
            )

        self.position_embedding = nn.Embedding(
            self.max_seq_length + 1, self.hidden_size
        )
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

        self.dis_projection = nn.Linear(self.hidden_size, 1)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.contras_loss_weight = config["contrastive_loss_weight"]
        self.contras_loss_temp = config["contras_loss_temp"]
        self.contras_target = config["contras_target"]
        self.contras_k = config["contras_k"]
        self.con_sim = Similarity(temp=self.contras_loss_temp)
        self.con_loss_fct = nn.CrossEntropyLoss()

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            if self.ignore_pad:
                self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            else:
                self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

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
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        return output  # [B L H]

    def forwardLogits(self, item_seq, item_seq_len, item_emb, bidirectional=True):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(
            item_seq, bidirectional=bidirectional
        )

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        return output  # [B L H]

    def forwardCLS(self, item_seq, item_seq_len, condition, bidirectional=True):
        nitem_seq = torch.cat(
            (torch.ones((item_seq.size(0), 1)).to(self.device), item_seq), dim=1
        )

        position_ids = torch.arange(
            item_seq.size(1) + 1, dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(nitem_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        nitem_emb = torch.cat((condition, item_emb), dim=1)

        input_emb = nitem_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(
            nitem_seq, bidirectional=bidirectional
        )

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = output[:, 1:, :]
        return output  # [B L H]

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

        con_loss = torch.tensor(0)

        if self.contras_loss_weight != 0:
            seq_output_1 = self.forward(item_seq, item_seq_len)
            k = self.contras_k
            if self.contras_target == "avg":
                seqs = torch.zeros((item_seq.size(0), self.max_seq_length))
                for i, l in enumerate(item_seq_len):
                    seqs[i][-l:] = 1
                seqs = seqs.unsqueeze(-1).repeat(1, 1, self.hidden_size)
                ksum = item_seq_len.unsqueeze(-1).repeat(1, self.hidden_size)
                seq_logits = (seq_output * seqs).sum(dim=1) / ksum
                seq_logits_1 = (seq_output_1 * seqs).sum(dim=1) / ksum

                logits_0 = seq_logits.unsqueeze(1)
                logits_1 = seq_logits_1.unsqueeze(0)
                cos_sim = self.con_sim(logits_0, logits_1).view(-1, logits_0.size(0))
                labels = torch.arange(logits_0.size(0)).long().to(self.device)
                con_loss = self.con_loss_fct(cos_sim, labels)

            # use lask k avg seq hidden to calculate con loss
            if self.contras_target == "avgk":
                logits_0 = seq_output[:, -k:, :].mean(dim=1).unsqueeze(1)
                logits_1 = seq_output_1[:, -k:, :].mean(dim=1).unsqueeze(0)
                cos_sim = self.con_sim(logits_0, logits_1).view(-1, logits_0.size(0))
                labels = torch.arange(logits_0.size(0)).long().to(self.device)
                con_loss = self.con_loss_fct(cos_sim, labels)

            # use inter seq hidden for each position to calculate con loss
            if self.contras_target == "interk":
                logits_0 = seq_output[:, -k:, :].transpose(0, 1).unsqueeze(2)
                logits_1 = seq_output_1[:, -k:, :].transpose(0, 1).unsqueeze(1)
                cos_sim = self.con_sim(logits_0, logits_1).view(-1, seq_output.size(0))
                labels = torch.arange(seq_output.size(0)).long().to(self.device)
                labels_n = labels.repeat(k)
                con_loss = self.con_loss_fct(cos_sim, labels_n)

        # self.loss_type = 'CE'
        loss = self.cross_entropy(seq_output, item_labeln)
        return loss, self.contras_loss_weight * con_loss

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
        seq_output = self.forward(item_seq, item_seq_len)
        # seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        seq_output = seq_output[:, -1, :].squeeze(1)

        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
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
