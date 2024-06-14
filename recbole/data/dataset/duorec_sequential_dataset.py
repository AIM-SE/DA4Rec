import numpy as np
import torch

from recbole.data.dataset import Dataset
from recbole.data.interaction import Interaction
from recbole.utils.enum_type import FeatureType, FeatureSource

# hook
from data_augmentation.utils.parallel_utils import ParallelDA


class DuoRecSequentialDataset(Dataset):
    def __init__(self, config):
        self.max_item_list_len = config["MAX_ITEM_LIST_LENGTH"]
        self.item_list_length_field = config["ITEM_LIST_LENGTH_FIELD"]
        super().__init__(config)
        if config["benchmark_filename"] is not None:
            self._benchmark_presets()
            self.same_target_index, self.instances = self.semantic_augmentation()

    def semantic_augmentation(self):
        import os

        root = os.path.join(self.dataset_path)
        aug_path = os.path.join(root, "semantic_augmentation.npy")
        instances = ParallelDA.read_inter(
            os.path.join(root, f"{self.dataset_name}.train.inter")
        )
        # if os.path.exists(aug_path):
        #     same_target_index = np.load(aug_path, allow_pickle=True)
        # else:
        if True:
            same_target_index = []
            target_index = []

            for seqs in instances:
                target_index.append(int(seqs[-1]))
            # target_item = self.inter_feat["item_id"][target_index].numpy()
            target_item = np.array(target_index)
            for index, item_id in enumerate(target_item):
                all_index_same_id = np.where(target_item == item_id)[
                    0
                ]  # all index of a specific item id with self item
                delete_index = np.argwhere(all_index_same_id == index)
                all_index_same_id_wo_self = np.delete(all_index_same_id, delete_index)
                same_target_index.append(all_index_same_id_wo_self)
            # same_target_index = np.array(same_target_index)
            # np.save(aug_path, same_target_index)

        return same_target_index, instances

    def _benchmark_presets(self):
        list_suffix = self.config["LIST_SUFFIX"]
        for field in self.inter_feat:
            if field + list_suffix in self.inter_feat:
                list_field = field + list_suffix
                setattr(self, f"{field}_list_field", list_field)
        self.set_field_property(
            self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
        )
        self.inter_feat[self.item_list_length_field] = self.inter_feat[
            self.item_id_list_field
        ].agg(len)

    def inter_matrix(self, form="coo", value_field=None):
        """Get sparse matrix that describe interactions between user_id and item_id.
        Sparse matrix has shape (user_num, item_num).
        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.inter_feat[src, tgt]``.
        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.
        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not self.uid_field or not self.iid_field:
            raise ValueError(
                "dataset does not exist uid/iid, thus can not converted to sparse matrix."
            )

        l1_idx = self.inter_feat[self.item_list_length_field] == 1
        l1_inter_dict = self.inter_feat[l1_idx].interaction
        new_dict = {}
        list_suffix = self.config["LIST_SUFFIX"]
        candidate_field_set = set()
        for field in l1_inter_dict:
            if field != self.uid_field and field + list_suffix in l1_inter_dict:
                candidate_field_set.add(field)
                new_dict[field] = torch.cat(
                    [self.inter_feat[field], l1_inter_dict[field + list_suffix][:, 0]]
                )
            elif (not field.endswith(list_suffix)) and (
                field != self.item_list_length_field
            ):
                new_dict[field] = torch.cat(
                    [self.inter_feat[field], l1_inter_dict[field]]
                )
        local_inter_feat = Interaction(new_dict)
        return self._create_sparse_matrix(
            local_inter_feat, self.uid_field, self.iid_field, form, value_field
        )

    def build(self):
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args != "TO":
            raise ValueError(
                f"The ordering args for sequential recommendation has to be 'TO'"
            )

        return super().build()
