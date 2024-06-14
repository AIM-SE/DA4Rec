"""
recbole.data.dataloader.da_dataloader
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


class AugmentedDataLoader(TrainDataLoader):
    """:class:`AugmentedDataLoader` is used for performing on-the-fly augmentation. It will do data augmentation for the
               origin data. And its returned data contains the following:

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

    def __init__(self, config, dataset, sampler, shuffle=False, aug_pipeline=None):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.max_item_list_len = config["MAX_ITEM_LIST_LENGTH"]
        self.pipeline = aug_pipeline

        self.config = config
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    @property
    def pr_end(self):
        return len(self.uid_list)

    def _shuffle(self):
        new_index = torch.randperm(self.pr_end)
        self.uid_list = self.uid_list[new_index]
        self.item_list_index = self.item_list_index[new_index]
        self.target_index = self.target_index[new_index]
        self.item_list_length = self.item_list_length[new_index]

    def _on_the_fly_aug(self, cur_data):
        seqs_shape = cur_data["item_id_list"].shape
        seqs = cur_data["item_id_list"].numpy().tolist()
        lengths = cur_data["item_length"].numpy().tolist()

        aug_seqs, aug_lens = self.pipeline(seqs, lengths)

        # Collate to align with torch.Tensor((1024, 50))
        seq_len = seqs_shape[1]
        for i in range(len(aug_seqs)):
            orig = aug_seqs[i]
            orig += [0 for i in range(seq_len - len(orig))]
            aug_seqs[i] = orig

        cur_data["item_id_list"] = torch.from_numpy(np.array(aug_seqs, dtype=np.int64))
        cur_data["item_length"] = torch.from_numpy(np.array(aug_lens, dtype=np.int64))

        return cur_data

    def get_online_stat(self):
        return self.pipeline.get_stat()

    def collate_fn(self, index):
        index = np.array(index)
        data = self._dataset[index]
        transformed_data = self.transform(self._dataset, data)
        return self._on_the_fly_aug(transformed_data)
