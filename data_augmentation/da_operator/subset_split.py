import numpy as np


class SubsetSplit:
    def __init__(self, items):
        self.items = items

    def init(self, instances, timestamps, **kwargs):
        return

    def _mask(self, seq, dropout_prob):
        while True:
            mask = np.random.rand(len(seq)) > dropout_prob
            if np.any(mask):
                return mask

    def _apply(self, seq, mask):
        return np.array(seq)[mask].tolist()

    def forward(self, seq, ts, **kwargs):
        """
        seq:                    (Iterable) the interaction sequence.
        ts:                     (Iterable) the interaction timestamp sequence.
        =========
        **kwargs:
        dropout_prob:
        start_pos:              (int) the start position for subset split.
        end_pos:                (int) the end position for subset split.
        subset_split_n_times:   (int) the number of subset splitting for each sequence.
        """
        dropout_prob = kwargs["dropout_prob"]
        start_pos = kwargs["start_pos"]
        end_pos = kwargs["end_pos"]
        n_times = kwargs["subset_split_n_times"]

        if end_pos is None:
            end_pos = len(seq)
        else:
            end_pos = len(seq) + end_pos

        aug_seqs = []
        aug_ts = None if ts is None else []
        for _ in range(n_times):
            mask = self._mask(seq[start_pos:end_pos], dropout_prob)
            augmented_seq = self._apply(seq[start_pos:end_pos], mask)
            aug_seqs.append(augmented_seq + seq[end_pos:])

            if aug_ts is not None:
                aug_ts.append(self._apply(ts[start_pos:end_pos], mask) + ts[end_pos:])

        return aug_seqs, aug_ts
