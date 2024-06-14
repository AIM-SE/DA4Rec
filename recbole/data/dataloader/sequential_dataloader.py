"""
recbole.data.dataloader.sequential_dataloader
################################################
"""

import numpy as np
import torch
import random
import math

from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbole.data.dataloader.general_dataloader import TrainDataLoader
from recbole.data.interaction import Interaction, cat_interactions
from recbole.utils import FeatureSource, FeatureType, InputType


class SequentialDataLoader(TrainDataLoader):
    """:class:`SequentialDataLoader` is used for sequential model. It will do data augmentation for the origin data.
    And its returned data contains the following:

        - user id
        - history items list
        - history items' interaction time list
        - item to be predicted
        - the interaction time of item to be predicted
        - history list length
        - other interaction information of item to be predicted

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False, phase="train"):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.time_field = dataset.time_field
        self.max_item_list_len = config["MAX_ITEM_LIST_LENGTH"]
        self.real_time = config["real_time"]

        self.instances = dataset.instances
        self.pre_processed_data = None

        self.static_item_id_list = None
        self.static_item_length = None

        # semantic augmentation
        self.phase = phase
        if config["SSL_AUG"] == "DuoRec" and self.phase == "train":
            self.same_target_index = dataset.same_target_index

        self.config = config
        self.data_preprocess(dataset)
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def data_preprocess(self, dataset):
        """Do data augmentation before training/evaluation."""
        # used for DuoRec semantic positive sampling
        if self.config["SSL_AUG"] == "DuoRec":
            self.static_item_id_list = (
                dataset.inter_feat["item_id_list"].detach().clone()
            )
            self.static_item_length = dataset.inter_feat["item_length"].detach().clone()

    @property
    def pr_end(self):
        return len(self.uid_list)

    def _shuffle(self):
        self.pre_processed_data.shuffle()
        if self.config["SSL_AUG"] == "DuoRec" and self.phase == "train":
            self.same_target_index = self.same_target_index[
                self.pre_processed_data.index
            ]

    def aug_inter(self, inter):
        return self.duorec_aug(inter)

    def duorec_aug(self, cur_data):
        null_index = []
        sample_pos = []
        for i, uid in enumerate(cur_data["session_id"]):
            targets = self.same_target_index[uid - 1]
            # in case there is no same-target sequence
            # don't know why this happens since the filtering has been applied
            if len(targets) == 0:
                sample_pos.append(-1)
                null_index.append(i)
            else:
                sample_pos.append(np.random.choice(targets))
        sem_pos_seqs = self.static_item_id_list[sample_pos]
        sem_pos_lengths = self.static_item_length[sample_pos]
        if null_index:
            sem_pos_seqs[null_index] = cur_data["item_id_list"][null_index]
            sem_pos_lengths[null_index] = cur_data["item_length"][null_index]

        cur_data.update(
            Interaction({"sem_aug": sem_pos_seqs, "sem_aug_lengths": sem_pos_lengths})
        )
        return cur_data

    def cl4srec_aug(self, cur_data):
        def item_crop(seq, length, eta=0.6):
            num_left = math.floor(length * eta)
            crop_begin = random.randint(0, length - num_left)
            croped_item_seq = np.zeros(seq.shape[0])
            if crop_begin + num_left < seq.shape[0]:
                croped_item_seq[:num_left] = seq[crop_begin : crop_begin + num_left]
            else:
                croped_item_seq[:num_left] = seq[crop_begin:]
            return torch.tensor(croped_item_seq, dtype=torch.long), torch.tensor(
                num_left, dtype=torch.long
            )

        def item_mask(seq, length, gamma=0.3):
            num_mask = math.floor(length * gamma)
            mask_index = random.sample(range(length), k=num_mask)
            masked_item_seq = seq[:]
            masked_item_seq[mask_index] = (
                self.dataset.item_num
            )  # token 0 has been used for semantic masking
            return masked_item_seq, length

        def item_reorder(seq, length, beta=0.6):
            num_reorder = math.floor(length * beta)
            reorder_begin = random.randint(0, length - num_reorder)
            reordered_item_seq = seq[:]
            shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
            random.shuffle(shuffle_index)
            reordered_item_seq[reorder_begin : reorder_begin + num_reorder] = (
                reordered_item_seq[shuffle_index]
            )
            return reordered_item_seq, length

        seqs = cur_data["item_id_list"]
        lengths = cur_data["item_length"]

        aug_seq1 = []
        aug_len1 = []
        aug_seq2 = []
        aug_len2 = []
        for seq, length in zip(seqs, lengths):
            if length > 1:
                switch = random.sample(range(3), k=2)
            else:
                switch = [3, 3]
                aug_seq = seq
                aug_len = length
            if switch[0] == 0:
                aug_seq, aug_len = item_crop(seq, length)
            elif switch[0] == 1:
                aug_seq, aug_len = item_mask(seq, length)
            elif switch[0] == 2:
                aug_seq, aug_len = item_reorder(seq, length)

            aug_seq1.append(aug_seq)
            aug_len1.append(aug_len)

            if switch[1] == 0:
                aug_seq, aug_len = item_crop(seq, length)
            elif switch[1] == 1:
                aug_seq, aug_len = item_mask(seq, length)
            elif switch[1] == 2:
                aug_seq, aug_len = item_reorder(seq, length)

            aug_seq2.append(aug_seq)
            aug_len2.append(aug_len)

        cur_data.update(
            Interaction(
                {
                    "aug1": torch.stack(aug_seq1),
                    "aug_len1": torch.stack(aug_len1),
                    "aug2": torch.stack(aug_seq2),
                    "aug_len2": torch.stack(aug_len2),
                }
            )
        )
