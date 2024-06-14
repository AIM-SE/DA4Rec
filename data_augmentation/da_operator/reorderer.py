import importlib
import copy
import random

from data_augmentation.da_operator.croper import Croper


class Reorderer:
    def __init__(self, items):
        self.items = items
        self._pos_sampler = None

    def init(self, instances, timestamps, **kwargs):
        """
        instances:              (Iterable) the interaction sequence.
        timestamps:             (Iterable) the interaction timestamp sequence.

        pos:                    (str, choice in ['uniform']) position sampling method.
        """
        pos = kwargs["pos"]

        module_path = "data_augmentation.utils.distribution_sampler"
        if importlib.util.find_spec(module_path):
            module = importlib.import_module(module_path)
            self._pos_sampler = getattr(module, pos + "DistributionPositionSampler")()
        else:
            raise ValueError(
                f"Invalid argument 'pos'[{pos}], it must be one of 'uniform', 'popularity', 'distance'."
            )

        if self._pos_sampler is None:
            raise ValueError(
                f"Invalid argument 'pos'[{pos}], it must be one of 'uniform', 'popularity', 'distance'."
            )

    def forward(self, seq, ts, **kwargs):
        """
        seq:                    (Iterable) the interaction sequence.
        ts:                     (Iterable) the interaction timestamp sequence.
        =========
        **kwargs:
        start_pos:              (int) the start position for replacing.
        end_pos:                (int) the end position for replacing.
        reorder_nums:           (int) the number of reordered items.
        reorder_ratio:          (float) the ratio of reordered items.
        reorder_times:          (int) the number of reorder for each sequence

        pop_counter:            (np.array) the popularity counter. [popularity-position-sampling]

        sub_seq_length:         (int) the length of sub-sequence [Ti-reorder]
        reorder_time_sort:      (str, choice in ["maximum", "minimum"]) [Ti-reorder]
        """
        start_pos = kwargs["start_pos"]
        end_pos = kwargs["end_pos"]
        reorder_nums = kwargs["reorder_nums"]
        reorder_ratio = kwargs["reorder_ratio"]
        n_times = kwargs["reorder_n_times"]

        if len(seq) == 2:
            return [], []

        if end_pos is None:
            end_pos = len(seq)
        else:
            end_pos = len(seq) + end_pos

        reorder_nums = max(reorder_nums, int((end_pos + 1 - start_pos) * reorder_ratio))

        op_type = kwargs["operation_type"]

        if op_type == "reorder":
            aug_seqs = []
            for _ in range(n_times):
                # Note: the 'reorder_pos' is an offset to the 'start_pos'
                reorder_pos = list(
                    map(
                        lambda x: x + start_pos,
                        self._pos_sampler.sample(
                            seq[start_pos : end_pos + 1 - reorder_nums], n=1, **kwargs
                        ),
                    )
                )[0]
                shuffle_seq = copy.deepcopy(
                    seq[reorder_pos : reorder_pos + reorder_nums]
                )
                random.shuffle(shuffle_seq)
                reorder_seq = (
                    seq[0:reorder_pos] + shuffle_seq + seq[reorder_pos + reorder_nums :]
                )
                aug_seqs.append(reorder_seq)
            return aug_seqs, None
        elif op_type == "Ti-reorder":
            # For Ti-reorder, it first selects a subsequence and reorders the item with in the subsequence.
            sub_seq_length = kwargs["sub_seq_length"]
            assert kwargs["reorder_time_sort"] in ["maximum", "minimum"]
            if sub_seq_length > end_pos - start_pos:
                return [], []  # skip if the seq is too short
            elif sub_seq_length == end_pos - start_pos:
                aug_seq = seq[start_pos:end_pos]
                random.shuffle(aug_seq)
                aug_ts = copy.deepcopy(ts)
                return [aug_seq + seq[end_pos:]], [aug_ts]

            crop_args = {
                "operation_type": "Ti-crop",
                "pos": "time",
                "start_pos": kwargs["start_pos"],
                "end_pos": kwargs["end_pos"] - 1,
                "crop_nums": sub_seq_length,
                "crop_ratio": 0,
                "crop_time_sort": kwargs["reorder_time_sort"],
                "crop_n_times": None,
                "ti_crop_n_times": 1,
                "ti_threshold": kwargs["ti_threshold"],
            }
            # The subsequences here are already sorted in descending order according to the std of time intervals.
            sub_item_seqs, sub_ts_seqs = Croper(self.items).forward(
                seq, ts, **crop_args
            )

            sub_seq = sub_item_seqs[0]
            # Find the start index of the sub_seq in the original sequence.
            for pos in range(start_pos, end_pos):
                if sub_seq == seq[pos : pos + len(sub_seq)]:
                    random.shuffle(sub_seq)
                    reorder_seq = (
                        seq[start_pos:pos] + sub_seq + seq[pos + len(sub_seq) :]
                    )
                    return [reorder_seq], [copy.deepcopy(ts)]
            return [], []
        else:
            raise ValueError(f"Invalid operation [{op_type}]")
