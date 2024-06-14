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
import math
import copy
import random
import itertools


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


def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)


class CombinatorialEnumerate(object):
    """Given M type of augmentations, and a original sequence, successively call \
    the augmentation 2*C(M, 2) times can generate total C(M, 2) augmentaion pairs.
    In another word, the augmentation method pattern will repeat after every 2*C(M, 2) calls.

    For example, M = 3, the argumentation methods to be called are in following order:
    a1, a2, a1, a3, a2, a3. Which formed three pair-wise augmentations:
    (a1, a2), (a1, a3), (a2, a3) for multi-view contrastive learning.
    """

    def __init__(
        self,
        tao=0.2,
        gamma=0.7,
        beta=0.2,
        item_similarity_model=None,
        insert_rate=0.3,
        max_insert_num_per_pos=3,
        substitute_rate=0.3,
        n_views=5,
    ):
        self.data_augmentation_methods = [
            Crop(tao=tao),
            Mask(gamma=gamma),
            Reorder(beta=gamma),
            Insert(
                item_similarity_model,
                insert_rate=insert_rate,
                max_insert_num_per_pos=max_insert_num_per_pos,
            ),
            Substitute(item_similarity_model, substitute_rate=substitute_rate),
        ]
        self.n_views = n_views
        # length of the list == C(M, 2)
        self.augmentation_idx_list = self.__get_augmentation_idx_order()
        self.total_augmentation_samples = len(self.augmentation_idx_list)
        self.cur_augmentation_idx_of_idx = 0

    def __get_augmentation_idx_order(self):
        augmentation_idx_list = []
        for view_1, view_2 in itertools.combinations(
            [i for i in range(self.n_views)], 2
        ):
            augmentation_idx_list.append(view_1)
            augmentation_idx_list.append(view_2)
        return augmentation_idx_list

    def __call__(self, sequence):
        augmentation_idx = self.augmentation_idx_list[self.cur_augmentation_idx_of_idx]
        augment_method = self.data_augmentation_methods[augmentation_idx]
        # keep the index of index in range(0, C(M,2))
        self.cur_augmentation_idx_of_idx += 1
        self.cur_augmentation_idx_of_idx = (
            self.cur_augmentation_idx_of_idx % self.total_augmentation_samples
        )
        # print(augment_method.__class__.__name__)
        return augment_method(sequence)


class Random(object):
    """Randomly pick one data augmentation type every time call"""

    def __init__(
        self,
        tao=0.2,
        gamma=0.7,
        beta=0.2,
        item_similarity_model=None,
        insert_rate=0.3,
        max_insert_num_per_pos=3,
        substitute_rate=0.3,
        augment_threshold=-1,
        augment_type_for_short="SIM",
    ):
        self.augment_threshold = augment_threshold
        self.augment_type_for_short = augment_type_for_short
        if self.augment_threshold == -1:
            self.data_augmentation_methods = [
                Crop(tao=tao),
                Mask(gamma=gamma),
                Reorder(beta=beta),
                Insert(
                    item_similarity_model,
                    insert_rate=insert_rate,
                    max_insert_num_per_pos=max_insert_num_per_pos,
                ),
                Substitute(item_similarity_model, substitute_rate=substitute_rate),
            ]
            # print("[CoSeRec] Total augmentation numbers: ", len(self.data_augmentation_methods))
        elif self.augment_threshold > 0:
            # print("[CoSeRec] short sequence augment type:", self.augment_type_for_short)
            if self.augment_type_for_short == "SI":
                self.short_seq_data_aug_methods = [
                    Insert(
                        item_similarity_model,
                        insert_rate=insert_rate,
                        max_insert_num_per_pos=max_insert_num_per_pos,
                        augment_threshold=self.augment_threshold,
                    ),
                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                ]
            elif self.augment_type_for_short == "SIM":
                self.short_seq_data_aug_methods = [
                    Insert(
                        item_similarity_model,
                        insert_rate=insert_rate,
                        max_insert_num_per_pos=max_insert_num_per_pos,
                        augment_threshold=self.augment_threshold,
                    ),
                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                    Mask(gamma=gamma),
                ]

            elif self.augment_type_for_short == "SIR":
                self.short_seq_data_aug_methods = [
                    Insert(
                        item_similarity_model,
                        insert_rate=insert_rate,
                        max_insert_num_per_pos=max_insert_num_per_pos,
                        augment_threshold=self.augment_threshold,
                    ),
                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                    Reorder(beta=gamma),
                ]
            elif self.augment_type_for_short == "SIC":
                self.short_seq_data_aug_methods = [
                    Insert(
                        item_similarity_model,
                        insert_rate=insert_rate,
                        max_insert_num_per_pos=max_insert_num_per_pos,
                        augment_threshold=self.augment_threshold,
                    ),
                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                    Crop(tao=tao),
                ]
            elif self.augment_type_for_short == "SIMR":
                self.short_seq_data_aug_methods = [
                    Insert(
                        item_similarity_model,
                        insert_rate=insert_rate,
                        max_insert_num_per_pos=max_insert_num_per_pos,
                        augment_threshold=self.augment_threshold,
                    ),
                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                    Mask(gamma=gamma),
                    Reorder(beta=gamma),
                ]
            elif self.augment_type_for_short == "SIMC":
                self.short_seq_data_aug_methods = [
                    Insert(
                        item_similarity_model,
                        insert_rate=insert_rate,
                        max_insert_num_per_pos=max_insert_num_per_pos,
                        augment_threshold=self.augment_threshold,
                    ),
                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                    Mask(gamma=gamma),
                    Crop(tao=tao),
                ]
            elif self.augment_type_for_short == "SIRC":
                self.short_seq_data_aug_methods = [
                    Insert(
                        item_similarity_model,
                        insert_rate=insert_rate,
                        max_insert_num_per_pos=max_insert_num_per_pos,
                        augment_threshold=self.augment_threshold,
                    ),
                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                    Reorder(beta=gamma),
                    Crop(tao=tao),
                ]
            else:
                # print("[CoSeRec] all aug set for short sequences")
                self.short_seq_data_aug_methods = [
                    Insert(
                        item_similarity_model,
                        insert_rate=insert_rate,
                        max_insert_num_per_pos=max_insert_num_per_pos,
                        augment_threshold=self.augment_threshold,
                    ),
                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                    Crop(tao=tao),
                    Mask(gamma=gamma),
                    Reorder(beta=gamma),
                ]
            self.long_seq_data_aug_methods = [
                Insert(
                    item_similarity_model,
                    insert_rate=insert_rate,
                    max_insert_num_per_pos=max_insert_num_per_pos,
                    augment_threshold=self.augment_threshold,
                ),
                Crop(tao=tao),
                Mask(gamma=gamma),
                Reorder(beta=gamma),
                Substitute(item_similarity_model, substitute_rate=substitute_rate),
            ]
            # print(
            #     # "[CoSeRec] Augmentation methods for Long sequences:",
            #     len(self.long_seq_data_aug_methods),
            # )
            # print(
            #     # "[CoSeRec] Augmentation methods for short sequences:",
            #     len(self.short_seq_data_aug_methods),
            # )
        else:
            raise ValueError("Invalid data type.")

    def __call__(self, sequence):
        if self.augment_threshold == -1:
            # randint generate int x in range: a <= x <= b
            augment_method_idx = random.randint(
                0, len(self.data_augmentation_methods) - 1
            )
            augment_method = self.data_augmentation_methods[augment_method_idx]
            # print(augment_method.__class__.__name__) # debug usage
            return augment_method(sequence)
        elif self.augment_threshold > 0:
            seq_len = len(sequence)
            if seq_len > self.augment_threshold:
                # randint generate int x in range: a <= x <= b
                augment_method_idx = random.randint(
                    0, len(self.long_seq_data_aug_methods) - 1
                )
                augment_method = self.long_seq_data_aug_methods[augment_method_idx]
                # print(augment_method.__class__.__name__) # debug usage
                return augment_method(sequence)
            elif seq_len <= self.augment_threshold:
                # randint generate int x in range: a <= x <= b
                augment_method_idx = random.randint(
                    0, len(self.short_seq_data_aug_methods) - 1
                )
                augment_method = self.short_seq_data_aug_methods[augment_method_idx]
                # print(augment_method.__class__.__name__) # debug usage
                return augment_method(sequence)


def _ensmeble_sim_models(top_k_one, top_k_two):
    # only support top k = 1 case so far
    #     print("offline: ",top_k_one, "online: ", top_k_two)
    if top_k_one[0][1] >= top_k_two[0][1]:
        return [top_k_one[0][0]]
    else:
        return [top_k_two[0][0]]


class Insert(object):
    """Insert similar items every time call"""

    def __init__(
        self,
        item_similarity_model,
        insert_rate=0.4,
        max_insert_num_per_pos=1,
        augment_threshold=14,
    ):
        self.augment_threshold = augment_threshold
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.insert_rate = insert_rate
        self.max_insert_num_per_pos = max_insert_num_per_pos

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        insert_nums = max(int(self.insert_rate * len(copied_sequence)), 1)
        if len(copied_sequence) == 0:
            return []
        insert_idx = random.sample(
            [i for i in range(len(copied_sequence))], k=insert_nums
        )
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                top_k = random.randint(
                    1, max(1, int(self.max_insert_num_per_pos / insert_nums))
                )
                if self.ensemble:
                    top_k_one = self.item_sim_model_1.most_similar(
                        item, top_k=top_k, with_score=True
                    )
                    top_k_two = self.item_sim_model_2.most_similar(
                        item, top_k=top_k, with_score=True
                    )
                    inserted_sequence += _ensmeble_sim_models(top_k_one, top_k_two)
                else:
                    inserted_sequence += self.item_similarity_model.most_similar(
                        item, top_k=top_k
                    )
            inserted_sequence += [item]

        return inserted_sequence


class Substitute(object):
    """Substitute with similar items"""

    def __init__(self, item_similarity_model, substitute_rate=0.1):
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.substitute_rate = substitute_rate

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        substitute_nums = max(int(self.substitute_rate * len(copied_sequence)), 1)
        if len(copied_sequence) == 0:
            return []
        # print(len(copied_sequence), substitute_nums)
        substitute_idx = random.sample(
            [i for i in range(len(copied_sequence))], k=substitute_nums
        )
        inserted_sequence = []
        for index in substitute_idx:
            if self.ensemble:
                top_k_one = self.item_sim_model_1.most_similar(
                    copied_sequence[index], with_score=True
                )
                top_k_two = self.item_sim_model_2.most_similar(
                    copied_sequence[index], with_score=True
                )
                substitute_items = _ensmeble_sim_models(top_k_one, top_k_two)
                copied_sequence[index] = substitute_items[0]
            else:
                copied_sequence[index] = copied_sequence[index] = (
                    self.item_similarity_model.most_similar(copied_sequence[index])[0]
                )
        return copied_sequence


class Crop(object):
    """Randomly crop a subseq from the original sequence"""

    def __init__(self, tao=0.2):
        self.tao = tao

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.tao * len(copied_sequence))
        if len(copied_sequence) == 0:
            return []
        # randint generate int x in range: a <= x <= b
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
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
        if len(copied_sequence) == 0:
            return []
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
        sub_seq_length = int(self.beta * len(copied_sequence))
        # print("[CoSeRec]", len(copied_sequence), sub_seq_length)
        if len(copied_sequence) == 0:
            start_index = 0
        else:
            start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        sub_seq = copied_sequence[start_index : start_index + sub_seq_length]
        random.shuffle(sub_seq)
        reordered_seq = (
            copied_sequence[:start_index]
            + sub_seq
            + copied_sequence[start_index + sub_seq_length :]
        )
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq


class CoSeRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(CoSeRec, self).__init__(config, dataset)
        args = config
        self.args = args
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

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        # offline_similarity_model = OfflineItemSimilarity(data_file=args.data_file,
        #                                                  similarity_path=args.similarity_model_path,
        #                                                  model_name=args.similarity_model_name,
        #                                                  dataset_name=args.data_name)

        # -----------   online based on shared item embedding for item similarity --------- #
        self.online_similarity_model = OnlineItemSimilarity(
            item_size=self.n_items, device=self.device
        )
        self.online_similarity_model.update_embedding_matrix(self.item_embedding)
        # self.online_similarity_model.update_embedding_matrix(self.item_embedding.cuda())
        similarity_model_type = config["similarity_model_type"]
        if similarity_model_type == "offline":
            self.similarity_model = self.offline_similarity_model
        elif similarity_model_type == "online":
            self.similarity_model = self.online_similarity_model
        elif similarity_model_type == "hybrid":
            self.similarity_model = [
                self.offline_similarity_model,
                self.online_similarity_model,
            ]
        # print("[CoSeRec] Similarity Model Type:", similarity_model_type)
        self.augmentations = {
            "crop": Crop(tao=args.tao),
            "mask": Mask(gamma=args.gamma),
            "reorder": Reorder(beta=args.beta),
            "substitute": Substitute(
                self.similarity_model, substitute_rate=args.substitute_rate
            ),
            "insert": Insert(
                self.similarity_model,
                insert_rate=args.insert_rate,
                max_insert_num_per_pos=args.max_insert_num_per_pos,
            ),
            "random": Random(
                tao=args.tao,
                gamma=args.gamma,
                beta=args.beta,
                item_similarity_model=self.similarity_model,
                insert_rate=args.insert_rate,
                max_insert_num_per_pos=args.max_insert_num_per_pos,
                substitute_rate=args.substitute_rate,
                augment_threshold=args.augment_threshold,
                augment_type_for_short=args.augment_type_for_short,
            ),
            "combinatorial_enumerate": CombinatorialEnumerate(
                tao=args.tao,
                gamma=args.gamma,
                beta=args.beta,
                item_similarity_model=self.similarity_model,
                insert_rate=args.insert_rate,
                max_insert_num_per_pos=args.max_insert_num_per_pos,
                substitute_rate=args.substitute_rate,
                n_views=args.n_views,
            ),
        }
        self.augment_type = args["augment_type"]
        self.base_transform = self.augmentations[self.augment_type]

        self.max_len = args["MAX_ITEM_LIST_LENGTH"]
        self.n_views = args["n_views"]

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

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.cf_criterion = NCELoss(self.args.temperature, self.device)

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
        subsequent_mask = subsequent_mask.long().to(self.device)

        # if self.args.use_gpu:
        #     # subsequent_mask = subsequent_mask.cuda()
        #     subsequent_mask = subsequent_mask.to(self.device)

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
            batch_rec_tensors.append(cur_rec_tensors)
            batch_cf_tensors.append(cf_tensors_list)
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

        return batch_rec_tensors, batch_cf_tensors

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

    def calculate_loss(self, interaction):
        pos_items = interaction[self.POS_ITEM_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        rec_batch, cl_batches = self._generate_inputs(interaction)
        rec_batch = tuple(t.to(self.device) for t in rec_batch)
        _, input_ids, target_pos, target_neg, _ = rec_batch

        # ---------- recommendation task ---------------#
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        rec_loss = self.loss_fct(logits, pos_items)
        # exit('debug')

        # ---------- contrastive learning task -------------#
        cl_losses = []
        for cl_batch in cl_batches:
            cl_loss = self._one_pair_contrastive_learning(cl_batch)
            cl_losses.append(cl_loss)
        joint_loss = self.args.rec_weight * rec_loss
        for cl_loss in cl_losses:
            joint_loss += self.args.cf_weight * cl_loss

        return joint_loss

    def _one_pair_contrastive_learning(self, inputs):
        """
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        """
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.forward(cl_batch)
        # cf_sequence_output = cf_sequence_output[:, -1, :]
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0] // 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1])
        return cl_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores


class OnlineItemSimilarity:
    def __init__(self, item_size, device):
        self.item_size = item_size
        self.item_embeddings = None
        self.device = device
        self.total_item_list = torch.tensor(
            [i for i in range(self.item_size)], dtype=torch.long
        ).to(self.device)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()

    def update_embedding_matrix(self, item_embeddings):
        self.item_embeddings = copy.deepcopy(item_embeddings).to(self.device)
        self.base_embedding_matrix = self.item_embeddings(self.total_item_list)

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item_idx in range(1, self.item_size):
            try:
                item_vector = self.item_embeddings(item_idx).view(-1, 1)
                item_similarity = torch.mm(
                    self.base_embedding_matrix, item_vector
                ).view(-1)
                max_score = max(torch.max(item_similarity), max_score)
                min_score = min(torch.min(item_similarity), min_score)
            except:
                continue
        return max_score, min_score

    def most_similar(self, item_idx, top_k=1, with_score=False):
        item_idx = torch.tensor(item_idx, dtype=torch.long).to(self.device)
        item_vector = self.item_embeddings(item_idx).view(-1, 1)
        item_similarity = torch.mm(self.base_embedding_matrix, item_vector).view(-1)
        item_similarity = (self.max_score - item_similarity) / (
            self.max_score - self.min_score
        )
        # remove item idx itself
        values, indices = item_similarity.topk(top_k + 1)
        if with_score:
            item_list = indices.tolist()
            score_list = values.tolist()
            if item_idx in item_list:
                idd = item_list.index(item_idx)
                item_list.remove(item_idx)
                score_list.pop(idd)
            return list(zip(item_list, score_list))
        item_list = indices.tolist()
        if item_idx in item_list:
            item_list.remove(item_idx)
        return item_list


class OfflineItemSimilarity:
    def __init__(
        self,
        data_file=None,
        similarity_path=None,
        model_name="ItemCF",
        dataset_name="Sports_and_Outdoors",
    ):
        self.dataset_name = dataset_name
        self.similarity_path = similarity_path
        # train_data_list used for item2vec, train_data_dict used for itemCF and itemCF-IUF
        (
            self.train_data_list,
            self.train_item_list,
            self.train_data_dict,
        ) = self._load_train_data(data_file)
        self.model_name = model_name
        self.similarity_model = self.load_similarity_model(self.similarity_path)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item in self.similarity_model.keys():
            for neig in self.similarity_model[item]:
                sim_score = self.similarity_model[item][neig]
                max_score = max(max_score, sim_score)
                min_score = min(min_score, sim_score)
        return max_score, min_score

    def _convert_data_to_dict(self, data):
        """
        split the data set
        testdata is a test data set
        traindata is a train set
        """
        train_data_dict = {}
        for user, item, record in data:
            train_data_dict.setdefault(user, {})
            train_data_dict[user][item] = record
        return train_data_dict

    def _save_dict(self, dict_data, save_path="./similarity.pkl"):
        # print("[CoSeRec] saving data to ", save_path)
        with open(save_path, "wb") as write_file:
            pickle.dump(dict_data, write_file)

    def _load_train_data(self, data_file=None):
        """
        read the data from the data file which is a data set
        """
        train_data = []
        train_data_list = []
        train_data_set_list = []
        for line in open(data_file).readlines():
            userid, items = line.strip().split(" ", 1)
            # only use training data
            items = items.split(" ")[:-3]
            train_data_list.append(items)
            train_data_set_list += items
            for itemid in items:
                train_data.append((userid, itemid, int(1)))
        return (
            train_data_list,
            set(train_data_set_list),
            self._convert_data_to_dict(train_data),
        )

    def _generate_item_similarity(self, train=None, save_path="./"):
        """
        calculate co-rated users between items
        """
        # print("[CoSeRec] getting item similarity...")
        train = train or self.train_data_dict
        C = dict()
        N = dict()

        if self.model_name in ["ItemCF", "ItemCF_IUF"]:
            # print("[CoSeRec] Step 1: Compute Statistics")
            for idx, (u, items) in enumerate(train.items()):
                if self.model_name == "ItemCF":
                    for i in items.keys():
                        N.setdefault(i, 0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i, {})
                            C[i].setdefault(j, 0)
                            C[i][j] += 1
                elif self.model_name == "ItemCF_IUF":
                    for i in items.keys():
                        N.setdefault(i, 0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i, {})
                            C[i].setdefault(j, 0)
                            C[i][j] += 1 / math.log(1 + len(items) * 1.0)
            self.itemSimBest = dict()
            # print("[CoSeRec] Step 2: Compute co-rate matrix")
            for idx, (cur_item, related_items) in enumerate(C.items()):
                self.itemSimBest.setdefault(cur_item, {})
                for related_item, score in related_items.items():
                    self.itemSimBest[cur_item].setdefault(related_item, 0)
                    self.itemSimBest[cur_item][related_item] = score / math.sqrt(
                        N[cur_item] * N[related_item]
                    )
            self._save_dict(self.itemSimBest, save_path=save_path)
        elif self.model_name == "Item2Vec":
            # details here: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py
            # print("[CoSeRec] Step 1: train item2vec model")
            item2vec_model = gensim.models.Word2Vec(
                sentences=self.train_data_list,
                vector_size=20,
                window=5,
                min_count=0,
                epochs=100,
            )
            self.itemSimBest = dict()
            total_item_nums = len(item2vec_model.wv.index_to_key)
            # print("[CoSeRec] Step 2: convert to item similarity dict")
            for cur_item in item2vec_model.wv.index_to_key:
                related_items = item2vec_model.wv.most_similar(
                    positive=[cur_item], topn=20
                )
                self.itemSimBest.setdefault(cur_item, {})
                for related_item, score in related_items:
                    self.itemSimBest[cur_item].setdefault(related_item, 0)
                    self.itemSimBest[cur_item][related_item] = score
            # print("[CoSeRec] Item2Vec model saved to: ", save_path)
            self._save_dict(self.itemSimBest, save_path=save_path)
        elif self.model_name == "LightGCN":
            # train a item embedding from lightGCN model, and then convert to sim dict
            # print("[CoSeRec] generating similarity model..")
            itemSimBest = light_gcn.generate_similarity_from_light_gcn(
                self.dataset_name
            )
            # print("[CoSeRec] LightGCN based model saved to: ", save_path)
            self._save_dict(itemSimBest, save_path=save_path)

    def load_similarity_model(self, similarity_model_path):
        if not similarity_model_path:
            raise ValueError("invalid path")
        elif not os.path.exists(similarity_model_path):
            # print("[CoSeRec] the similirity dict not exist, generating...")
            self._generate_item_similarity(save_path=self.similarity_path)
        if self.model_name in ["ItemCF", "ItemCF_IUF", "Item2Vec", "LightGCN"]:
            with open(similarity_model_path, "rb") as read_file:
                similarity_dict = pickle.load(read_file)
            return similarity_dict
        elif self.model_name == "Random":
            similarity_dict = self.train_item_list
            return similarity_dict

    def most_similar(self, item, top_k=1, with_score=False):
        if self.model_name in ["ItemCF", "ItemCF_IUF", "Item2Vec", "LightGCN"]:
            """TODO: handle case that item not in keys"""
            if str(item) in self.similarity_model:
                top_k_items_with_score = sorted(
                    self.similarity_model[str(item)].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[0:top_k]
                if with_score:
                    return list(
                        map(
                            lambda x: (
                                int(x[0]),
                                (self.max_score - float(x[1]))
                                / (self.max_score - self.min_score),
                            ),
                            top_k_items_with_score,
                        )
                    )
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            elif int(item) in self.similarity_model:
                top_k_items_with_score = sorted(
                    self.similarity_model[int(item)].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[0:top_k]
                if with_score:
                    return list(
                        map(
                            lambda x: (
                                int(x[0]),
                                (self.max_score - float(x[1]))
                                / (self.max_score - self.min_score),
                            ),
                            top_k_items_with_score,
                        )
                    )
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            else:
                item_list = list(self.similarity_model.keys())
                random_items = random.sample(item_list, k=top_k)
                if with_score:
                    return list(map(lambda x: (int(x), 0.0), random_items))
                return list(map(lambda x: int(x), random_items))
        elif self.model_name == "Random":
            random_items = random.sample(self.similarity_model, k=top_k)
            if with_score:
                return list(map(lambda x: (int(x), 0.0), random_items))
            return list(map(lambda x: int(x), random_items))
