import copy
import importlib


class Croper:
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
        start_pos:              (int) the start position for crop.
        end_pos:                (int) the end position for crop.
        crop_nums:              (int) the size of cropping.
        crop_ratio:             (float) the ratio of cropping.
        crop_n_times:           (int) the number of cropping
        ti_crop_n_times:        (int) the number of Ti-crop

        pop_counter:            (np.array) the popularity counter. [popularity-position-sampling]

        crop_time_sort:         (str, choice in ["maximum", "minimum"]) [Ti-crop-position-sampling]
        """
        start_pos = kwargs["start_pos"]
        end_pos = kwargs["end_pos"]
        crop_ratio = kwargs["crop_ratio"]
        crop_nums = kwargs["crop_nums"]
        n_times = kwargs["crop_n_times"]
        ti_n_times = kwargs["ti_crop_n_times"]
        threshold = kwargs["ti_threshold"]

        if len(seq) == 2:
            return [], []

        if end_pos is None:
            end_pos = len(seq)
        else:
            end_pos = len(seq) + end_pos

        crop_nums = max(crop_nums, int((end_pos + 1 - start_pos) * crop_ratio))

        if start_pos + crop_nums > end_pos + 1:
            raise ValueError(
                f"'start_pos + crop_nums'[{start_pos + crop_nums}] must be smaller than 'end_pos'[{end_pos}]."
            )

        op_type = kwargs["operation_type"]

        if op_type == "crop":
            candidates = (
                seq[start_pos : end_pos + 2 - crop_nums]
                if start_pos + crop_nums < end_pos + 1
                else seq[start_pos]
            )
            # Note: the 'crop_pos' is an offset to the 'start_pos'
            crop_pos = list(
                map(
                    lambda x: x + start_pos,
                    self._pos_sampler.sample(candidates, n=n_times, **kwargs),
                )
            )
            crops = []
            for pos in crop_pos:
                crops.append(seq[pos : pos + crop_nums])
            return crops, None
        elif op_type == "Ti-crop":
            # For time-interval-aware crop, it's necessary to pass the whole slice ([start_pos: end_pos + 1]).
            if end_pos + 1 - start_pos < threshold:
                return [], []  # skip when the size is smaller than the threshold
            elif end_pos + 1 - start_pos == threshold:
                return [seq[start_pos : end_pos + 1]], [ts[start_pos : end_pos + 1]]

            crop_nums = threshold
            params = kwargs.copy()
            params["crop_nums"] = crop_nums
            crop_pos = list(
                map(
                    lambda x: x + start_pos,
                    self._pos_sampler.sample(
                        ts[start_pos : end_pos + 1], n=ti_n_times, **params
                    ),
                )
            )
            crop_inters = []
            crop_ts = []
            for pos in crop_pos:
                crop_inters.append(seq[pos : pos + crop_nums])
                crop_ts.append(ts[pos : pos + crop_nums])
            return crop_inters, crop_ts
        else:
            raise ValueError(f"Invalid operation [{op_type}]")
