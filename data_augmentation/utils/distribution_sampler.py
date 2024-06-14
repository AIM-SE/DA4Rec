import random
import math
import pickle
import os
import numpy as np


class AbstractDistributionPositionSampler(object):
    def __init__(self):
        pass

    def set_properties(self, **kwargs):
        for key, value in kwargs:
            self.__setattr__(key, value)

    def sample(self, seq, n: int, **kwargs):
        raise NotImplementedError(
            "Sampling method is not implemented in the abstract class."
        )


class uniformDistributionPositionSampler(AbstractDistributionPositionSampler):
    def __init__(self):
        super().__init__()

    def sample(self, seq, n: int, **kwargs):
        """
        **kwargs:
        operation_type:     (str, choice in ['insert', 'delete', 'crop', 'mask', 'replace', 'reorder']) the operation type.
        """

        op_type = kwargs["operation_type"]

        candidate_pos = list(range(len(seq)))
        if op_type in ["insert", "crop", "replace", "reorder"]:
            return np.random.choice(candidate_pos, size=n, replace=True).tolist()
        elif op_type in ["delete", "mask"]:
            if n > len(candidate_pos):
                raise ValueError(
                    f"The number of sampling must be larger than the length of the sequence."
                )
            else:
                return np.random.choice(candidate_pos, size=n, replace=False).tolist()
        else:
            raise ValueError(
                f"The 'op_type'[{op_type}] must be one of 'insert' or 'delete'. "
            )


class popularityDistributionPositionSampler(AbstractDistributionPositionSampler):
    def __init__(self):
        super().__init__()

    def sample(self, seq, n: int, **kwargs):
        """
        **kwargs:
        operation_type:     (str, choice in ['insert', 'delete', 'crop', 'mask', 'replace', 'reorder']) the operation type.
        pop_counter:       (np.array) the popularity counter.
        """
        op_type = kwargs["operation_type"]
        item_counter = kwargs["pop_counter"]

        item_counts = [item_counter[each_item] for each_item in seq]
        total_count = sum(item_counts)
        item_probs = [count / total_count for count in item_counts]
        candidate_pos = list(range(len(seq)))

        if op_type in ["insert", "crop", "replace", "reorder"]:
            return np.random.choice(
                candidate_pos, p=item_probs, size=n, replace=True
            ).tolist()
        elif op_type in ["delete", "mask"]:
            return np.random.choice(
                candidate_pos, p=item_probs, size=n, replace=False
            ).tolist()
        else:
            raise ValueError(
                f"The 'op_type'[{op_type}] must be one of 'insert' or 'delete'. "
            )


class distanceDistributionPositionSampler(AbstractDistributionPositionSampler):
    def __init__(self):
        super().__init__()

    def sample(self, seq, n: int, **kwargs):
        """
        **kwargs:
        operation_type:     (str, choice in ['insert', 'delete', 'crop', 'mask', 'replace', 'reorder']) the operation type.
        """

        op_type = kwargs["operation_type"]

        if len(seq) == 1:
            return [0]

        distances = list(range(len(seq)))
        total_distance = sum(distances)
        distances_probs = [distance / total_distance for distance in distances]
        item_probs = distances_probs[::-1]
        candidate_pos = list(range(len(seq)))
        if op_type in ["insert", "crop", "replace", "reorder"]:
            return np.random.choice(
                candidate_pos, p=item_probs, size=n, replace=True
            ).tolist()
        elif op_type in ["delete", "mask"]:
            return np.random.choice(
                candidate_pos, p=item_probs, size=n, replace=False
            ).tolist()
        else:
            raise ValueError(
                f"The 'op_type'[{op_type}] must be one of 'insert' or 'delete'. "
            )


class timeDistributionPositionSampler(AbstractDistributionPositionSampler):
    def __init__(self):
        super().__init__()

    def sample(self, seq, n: int, **kwargs):
        """
        seq:                    (Iterable) the interaction timestamp sequence.
        =========
        **kwargs:
        operation_type:         (str)

        insert_time_sort:       (str, choice in ["maximum", "minimum"]) [Ti-insert]

        mask_time_sort:         (str, choice in ["maximum", "minimum"]) [Ti-mask]

        sub_seq_length:         (int)
        """
        # The 'seq' here is a timestamp sequence
        op_type = kwargs["operation_type"]

        if op_type == "Ti-crop":
            # Get the cropping size
            crop_nums = kwargs["crop_nums"]
            crop_ratio = kwargs["crop_ratio"]

            crop_nums = max(crop_nums, int(len(seq) * crop_ratio))

            if len(seq) == crop_nums:
                return [0]

            ti_list = self.get_time_interval_list(seq)

            candidate_pos = range(
                0, len(seq) + 1 - crop_nums
            )  # len(seq) + 1 - crop_nums > 1
            if n > len(candidate_pos):
                raise ValueError(
                    f"The candidate set size [{len(candidate_pos)}] must be smaller than 'n_time' [{n}]"
                )

            candidate_slice_stds = [
                np.std(ti_list[pos : pos + crop_nums + 1]) for pos in candidate_pos
            ]
            assert kwargs["crop_time_sort"] in ["maximum", "minimum"]
            return (
                np.argsort(candidate_slice_stds)[::-1][:n]
                if kwargs["crop_time_sort"] == "maximum"
                else np.argsort(candidate_slice_stds)[:n]
            )
        elif op_type == "Ti-insert":
            assert kwargs["insert_time_sort"] in ["maximum", "minimum"]
            ti_list = self.get_time_interval_list(seq)
            return (
                np.argsort(ti_list)[::-1][:n]
                if kwargs["insert_time_sort"] == "maximum"
                else np.argsort(ti_list)[:n]
            )
        elif op_type == "Ti-delete":
            assert n < len(seq)
            # The behavior of "Ti-delete" is just like "Ti-insert"
            assert kwargs["delete_time_sort"] in ["maximum", "minimum"]
            ti_list = self.get_time_interval_list(seq)
            return (
                np.argsort(ti_list)[::-1][:n]
                if kwargs["delete_time_sort"] == "maximum"
                else np.argsort(ti_list)[:n]
            )
        elif op_type == "Ti-mask":
            # The behavior of "Ti-mask" is just like "Ti-insert"
            assert kwargs["mask_time_sort"] in ["maximum", "minimum"]
            ti_list = self.get_time_interval_list(seq)
            return (
                np.argsort(ti_list)[::-1][:n]
                if kwargs["mask_time_sort"] == "maximum"
                else np.argsort(ti_list)[:n]
            )
        else:
            raise ValueError(f"Invalid operation type [{op_type}]")

    def get_time_interval_list(self, ts):
        return [ts[idx + 1] - ts[idx] for idx in range(len(ts) - 1)]


class AbstractItemSampler:
    def __init__(self, items):
        self.items = items

    def set_properties(self, **kwargs):
        for key, value in kwargs:
            self.__setattr__(key, value)

    def sample(self, seq, n_times: int, **kwargs):
        """Note: the argument 'seq' here is passed by reference."""
        raise NotImplementedError(
            "Sampling method is not implemented in the abstract class."
        )


class randomItemSampler(AbstractItemSampler):
    def __init__(self, items):
        super().__init__(items)

    def sample(self, seq, n_times: int, **kwargs):
        return random.choices(self.items, k=n_times)


class similarItemSampler(AbstractItemSampler):
    def __init__(self, items):
        super().__init__(items)

    def sample(self, seq, n_times: int, **kwargs):
        """
        **kwargs:
        target_item:        (int) the index of the target item.
        item_embeddings:    (martix) the embeddings of all items.
        op:                 (Callable) the similarity measurement function.
        """
        target_item = kwargs["target_item"]
        item_embeddings = kwargs["item_embeddings"]
        op = kwargs["op"]

        target_embedding = item_embeddings[target_item]  # [n_items, embedding_size]
        scores = op(target_embedding, item_embeddings).sum(axis=1)
        candidate = np.argsort(scores)[::-1]
        return candidate[1 : 1 + n_times]


class unvisitedItemSampler(AbstractItemSampler):
    def __init__(self, items):
        super().__init__(items)

    def sample(self, seq, n_times: int, **kwargs):
        unvisited = set(self.items) - set(seq)
        if len(unvisited) == 0:
            return random.choices(self.items, k=n_times)
        else:
            return random.choices(list(unvisited), k=n_times)


class redundantItemSampler(AbstractItemSampler):
    def __init__(self, items):
        super().__init__(items)

    def sample(self, n_times: int, **kwargs):
        """
        **kwargs:
        seq:        (Iterable) the interaction sequence.
        """

        seq = kwargs["seq"]

        return random.choices(seq, k=n_times)


class memorybasedItemSampler(AbstractItemSampler):
    def __init__(self, items):
        super().__init__(items)
        self.items = items
        self.model_name = "ItemCF_IUF"
        self._train_data_list = []
        self._train_item_list = None
        self._train_data_dict = None
        self._item_sim_best = None
        self._similarity_model_path = None
        self._similarity_model = None
        self._max_score = None
        self._min_score = None

    def set_properties(self, **kwargs):
        for key, value in kwargs:
            self.__setattr__(key, value)

    def init(self, instances, **kwargs):
        self.model_name = kwargs["mb_model_name"]
        save_path = os.path.join(kwargs["tempdir"], kwargs["task_name"])
        os.makedirs(save_path, exist_ok=True)
        self._similarity_model_path = os.path.join(save_path, "similarity.pkl")

        user_id = 0
        train_data = []
        train_data_set_list = []
        for seq in instances:
            user_id += 1
            self._train_data_list.append(seq)
            train_data_set_list += seq
            for itemid in seq:
                train_data.append((user_id, itemid, int(1)))

        self._train_item_list = set(train_data_set_list)
        self._train_data_dict = self._convert_data_to_dict(train_data)
        self._similarity_model = self._load_similarity_model(
            self._similarity_model_path
        )
        self._max_score, self._min_score = self._get_maximum_minimum_sim_scores()

    def _get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item in self._similarity_model.keys():
            for neig in self._similarity_model[item]:
                sim_score = self._similarity_model[item][neig]
                max_score = max(max_score, sim_score)
                min_score = min(min_score, sim_score)
        return max_score, min_score

    def _generate_item_similarity(self, save_path):
        """
        calculate co-rated users between items
        """
        train = self._train_data_dict
        C = dict()
        N = dict()

        if self.model_name in ["ItemCF", "ItemCF_IUF"]:
            # print("[memory-based] Step 1: Compute Statistics")
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
            self._item_sim_best = dict()
            # print("[memory-based] Step 2: Compute co-rate matrix")
            for idx, (cur_item, related_items) in enumerate(C.items()):
                self._item_sim_best.setdefault(cur_item, {})
                for related_item, score in related_items.items():
                    self._item_sim_best[cur_item].setdefault(related_item, 0)
                    self._item_sim_best[cur_item][related_item] = score / math.sqrt(
                        N[cur_item] * N[related_item]
                    )
            self._save_dict(self._item_sim_best, save_path=save_path)

    def _save_dict(self, dict_data, save_path):
        # print("[memory-based] saving data to ", save_path)
        with open(save_path, "wb") as write_file:
            pickle.dump(dict_data, write_file)

    def _load_similarity_model(self, similarity_model_path):
        if not os.path.exists(similarity_model_path):
            # print("[CoSeRec] the similirity dict not exist, generating...")
            self._generate_item_similarity(save_path=similarity_model_path)
        if self.model_name in ["ItemCF", "ItemCF_IUF"]:
            with open(similarity_model_path, "rb") as read_file:
                similarity_dict = pickle.load(read_file)
            return similarity_dict
        elif self.model_name == "Random":
            similarity_dict = self._train_item_list
            return similarity_dict

    def _convert_data_to_dict(self, data):
        """
        split the data set
        traindata is a train set
        """
        train_data_dict = {}
        for user, item, record in data:
            train_data_dict.setdefault(user, {})
            train_data_dict[user][item] = record
        return train_data_dict

    def most_similar(self, item, top_k=1, with_score=False):
        if self.model_name in ["ItemCF", "ItemCF_IUF"]:
            """TODO: handle case that item not in keys"""
            if str(item) in self._similarity_model:
                top_k_items_with_score = sorted(
                    self._similarity_model[str(item)].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[0:top_k]
                if with_score:
                    return list(
                        map(
                            lambda x: (
                                int(x[0]),
                                (self._max_score - float(x[1]))
                                / (self._max_score - self._min_score),
                            ),
                            top_k_items_with_score,
                        )
                    )
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            elif int(item) in self._similarity_model:
                top_k_items_with_score = sorted(
                    self._similarity_model[int(item)].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[0:top_k]
                if with_score:
                    return list(
                        map(
                            lambda x: (
                                int(x[0]),
                                (self._max_score - float(x[1]))
                                / (self._max_score - self._min_score),
                            ),
                            top_k_items_with_score,
                        )
                    )
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            else:
                item_list = list(self._similarity_model.keys())
                random_items = random.sample(item_list, k=top_k)
                if with_score:
                    return list(map(lambda x: (int(x), 0.0), random_items))
                return list(map(lambda x: int(x), random_items))
        elif self.model_name == "Random":
            random_items = random.sample(self._similarity_model, k=top_k)
            if with_score:
                return list(map(lambda x: (int(x), 0.0), random_items))
            return list(map(lambda x: int(x), random_items))

    def sample(self, seq, n_times: int, **kwargs):
        """Note: the argument 'seq' here is passed by reference."""
        insert_pos = kwargs["insert_pos"]
        assert len(insert_pos) == n_times

        selected_item = []
        for idx in insert_pos:
            next_item = seq[idx]
            most_similar_item = str(self.most_similar(item=next_item, top_k=1)[0])
            selected_item.append(most_similar_item)
        return selected_item
