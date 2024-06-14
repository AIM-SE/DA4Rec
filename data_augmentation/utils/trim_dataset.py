import os
import time
import pandas as pd
import numpy as np

from collections import Counter
from recbole.data.interaction import Interaction

from recbole.utils import (
    FeatureSource,
    FeatureType,
)


class TrimDataset(object):
    def __init__(self, config, token, dataset_path):
        print(f"Dataset initialization started")
        start_time = time.time()
        self.config = config

        self.field2type = {}
        self.field2source = {}
        self.field2id_token = {}
        self.field2token_id = {}
        self.field2bucketnum = {}
        self.field2seqlen = {}
        self.alias = {}

        self.uid_field = self.config["USER_ID_FIELD"]
        self.iid_field = self.config["ITEM_ID_FIELD"]
        self.label_field = self.config["LABEL_FIELD"]
        self.time_field = self.config["TIME_FIELD"]

        if (self.uid_field is None) ^ (self.iid_field is None):
            raise ValueError(
                "USER_ID_FIELD and ITEM_ID_FIELD need to be set at the same time or not set at the same time."
            )

        self._load_data(token, os.path.join(dataset_path, token))
        self._init_alias()
        self._data_processing()

        end_time = time.time()
        print(
            f"Dataset initialization completed, elapsed: {round(end_time-start_time, 4)} s",
            flush=True,
        )
        print(
            f"Remaining users: {self.user_num}, items: {self.item_num}, "
            f"interactions: {self.inter_num}, sparsity: {self.sparsity}",
            flush=True,
        )

    def _load_data(self, token, dataset_path):
        """Load features.
        Load interaction features, then user/item features optionally,
        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
        """
        self._load_inter_feat(token, dataset_path)
        self.user_feat = self._load_user_or_item_feat(
            token, dataset_path, FeatureSource.USER, "uid_field"
        )
        self.item_feat = self._load_user_or_item_feat(
            token, dataset_path, FeatureSource.ITEM, "iid_field"
        )

    def _load_inter_feat(self, token, dataset_path):
        """Load interaction features.
        Load interaction features from ``.inter``.
        After loading, ``self.file_size_list`` stores the length of each interaction file.
        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
        """
        inter_feat_path = os.path.join(dataset_path, f"{token}.inter")
        if not os.path.isfile(inter_feat_path):
            raise ValueError(f"File {inter_feat_path} not exist.")

        inter_feat = self._load_feat(inter_feat_path, FeatureSource.INTERACTION)
        self.inter_feat = inter_feat

    def _load_feat(self, filepath, source):
        """Load features according to source into :class:`pandas.DataFrame`.
        Set features' properties, e.g. type, source and length.
        Args:
            filepath (str): path of input file.
            source (FeatureSource or str): source of input file.
        Returns:
            pandas.DataFrame: Loaded feature
        Note:
            For sequence features, ``seqlen`` will be loaded, but data in DataFrame will not be cut off.
            Their length is limited only after calling :meth:`~_dict_to_interaction` or
            :meth:`~_dataframe_to_interaction`
        """
        load_col, unload_col = self._get_load_and_unload_col(source)
        if load_col == set():
            return None

        field_separator = self.config["field_separator"]
        columns = []
        usecols = []
        dtype = {}
        encoding = self.config["encoding"]
        with open(filepath, "r", encoding=encoding) as f:
            head = f.readline()[:-1]
        for field_type in head.split(field_separator):
            field, ftype = field_type.split(":")
            try:
                ftype = FeatureType(ftype)
            except ValueError:
                raise ValueError(f"Type {ftype} from field {field} is not supported.")
            if load_col is not None and field not in load_col:
                continue
            if unload_col is not None and field in unload_col:
                continue
            if isinstance(source, FeatureSource) or source != "link":
                self.field2source[field] = source
                self.field2type[field] = ftype
                if not ftype.value.endswith("seq"):
                    self.field2seqlen[field] = 1
                if "float" in ftype.value:
                    self.field2bucketnum[field] = 2
            columns.append(field)
            usecols.append(field_type)
            dtype[field_type] = np.float64 if ftype == FeatureType.FLOAT else str

        df = pd.read_csv(
            filepath,
            delimiter=field_separator,
            usecols=usecols,
            dtype=dtype,
            encoding=encoding,
            engine="python",
        )
        df.columns = columns

        seq_separator = self.config["seq_separator"]
        for field in columns:
            ftype = self.field2type[field]
            if not ftype.value.endswith("seq"):
                continue
            df[field].fillna(value="", inplace=True)
            if ftype == FeatureType.TOKEN_SEQ:
                df[field] = [
                    np.array(list(filter(None, _.split(seq_separator))))
                    for _ in df[field].values
                ]
            elif ftype == FeatureType.FLOAT_SEQ:
                df[field] = [
                    np.array(list(map(float, filter(None, _.split(seq_separator)))))
                    for _ in df[field].values
                ]
            max_seq_len = max(map(len, df[field].values))
            if self.config["seq_len"] and field in self.config["seq_len"]:
                seq_len = self.config["seq_len"][field]
                df[field] = [
                    seq[:seq_len] if len(seq) > seq_len else seq
                    for seq in df[field].values
                ]
                self.field2seqlen[field] = min(seq_len, max_seq_len)
            else:
                self.field2seqlen[field] = max_seq_len

        return df

    def _get_load_and_unload_col(self, source):
        """Parsing ``config['load_col']`` and ``config['unload_col']`` according to source.
        See :doc:`../../user_guide/config/data_settings` for detail arg setting.
        Args:
            source (FeatureSource): source of input file.
        Returns:
            tuple: tuple of parsed ``load_col`` and ``unload_col``, see :doc:`../../user_guide/data/data_args` for details.
        """
        if isinstance(source, FeatureSource):
            source = source.value
        if self.config["load_col"] is None:
            load_col = None
        elif source not in self.config["load_col"]:
            load_col = set()
        elif self.config["load_col"][source] == "*":
            load_col = None
        else:
            load_col = set(self.config["load_col"][source])

        if (
            self.config["unload_col"] is not None
            and source in self.config["unload_col"]
        ):
            unload_col = set(self.config["unload_col"][source])
        else:
            unload_col = None

        if load_col and unload_col:
            raise ValueError(
                f"load_col [{load_col}] and unload_col [{unload_col}] can not be set the same time."
            )

        return load_col, unload_col

    def _load_user_or_item_feat(self, token, dataset_path, source, field_name):
        """Load user/item features.
        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
            source (FeatureSource): source of user/item feature.
            field_name (str): ``uid_field`` or ``iid_field``
        Returns:
            pandas.DataFrame: Loaded feature
        Note:
            ``user_id`` and ``item_id`` has source :obj:`~recbole.utils.enum_type.FeatureSource.USER_ID` and
            :obj:`~recbole.utils.enum_type.FeatureSource.ITEM_ID`
        """
        feat_path = os.path.join(dataset_path, f"{token}.{source.value}")
        field = getattr(self, field_name, None)

        if os.path.isfile(feat_path):
            feat = self._load_feat(feat_path, source)
        else:
            feat = None

        if feat is not None and field is None:
            raise ValueError(
                f"{field_name} must be exist if {source.value}_feat exist."
            )
        if feat is not None and field not in feat:
            raise ValueError(
                f"{field_name} must be loaded if {source.value}_feat is loaded."
            )
        if feat is not None:
            feat.drop_duplicates(subset=[field], keep="first", inplace=True)

        if field in self.field2source:
            self.field2source[field] = FeatureSource(source.value + "_id")
        return feat

    def _data_processing(self):
        self.feat_name_list = self._build_feat_name_list()
        self._set_label_by_threshold()
        self._data_filtering()
        self._remap_ID_all()

    def _build_feat_name_list(self):
        """Feat list building.
        Any feat loaded by Dataset can be found in ``feat_name_list``
        Returns:
            built feature name list.
        Note:
            Subclasses can inherit this method to add new feat.
        """
        feat_name_list = [
            feat_name
            for feat_name in ["inter_feat", "user_feat", "item_feat"]
            if getattr(self, feat_name, None) is not None
        ]
        return feat_name_list

    def _data_filtering(self):
        """Data filtering
        - Filter missing user_id or item_id
        - Remove duplicated user-item interaction
        - Value-based data filtering
        - Remove interaction by user or item
        - K-core data filtering
        Note:
            After filtering, feats(``DataFrame``) has non-continuous index,
            thus :meth:`~recbole.data.dataset.dataset.Dataset._reset_index` will reset the index of feats.
        """
        self._filter_nan_user_or_item()
        self._remove_duplication()
        self._filter_by_field_value()
        self._filter_inter_by_user_or_item()
        self._filter_by_inter_num()
        self._reset_index()

    def _filter_nan_user_or_item(self):
        """Filter NaN user_id and item_id"""
        for field, name in zip([self.uid_field, self.iid_field], ["user", "item"]):
            feat = getattr(self, name + "_feat")
            if feat is not None:
                dropped_feat = feat.index[feat[field].isnull()]
                if len(dropped_feat):
                    print(
                        f"In {name}_feat, line {list(dropped_feat + 2)}, {field} do not exist, so they will be removed.",
                        flush=True,
                    )
                    feat.drop(feat.index[dropped_feat], inplace=True)
            if field is not None:
                dropped_inter = self.inter_feat.index[self.inter_feat[field].isnull()]
                if len(dropped_inter):
                    print(
                        f"In inter_feat, line {list(dropped_inter + 2)}, {field} do not exist, so they will be removed.",
                        flush=True,
                    )
                    self.inter_feat.drop(
                        self.inter_feat.index[dropped_inter], inplace=True
                    )

    def _remove_duplication(self):
        """Remove duplications in inter_feat.
        If :attr:`self.config['rm_dup_inter']` is not ``None``, it will remove duplicated user-item interactions.
        Note:
            Before removing duplicated user-item interactions, if :attr:`time_field` existed, :attr:`inter_feat`
            will be sorted by :attr:`time_field` in ascending order.
        """
        keep = self.config["rm_dup_inter"]
        if keep is None:
            return
        self._check_field("uid_field", "iid_field")

        if self.time_field in self.inter_feat:
            self.inter_feat.sort_values(
                by=[self.time_field], ascending=True, inplace=True
            )
            print(
                f"Records in original dataset have been sorted by value of [{self.time_field}] in ascending order.",
                flush=True,
            )
        else:
            print(
                f"Timestamp field has not been loaded or specified, "
                f"thus strategy [{keep}] of duplication removal may be meaningless.",
                flush=True,
            )
        self.inter_feat.drop_duplicates(
            subset=[self.uid_field, self.iid_field], keep=keep, inplace=True
        )

    def _filter_by_inter_num(self):
        """Filter by number of interaction.
        The interval of the number of interactions can be set, and only users/items whose number
        of interactions is in the specified interval can be retained.
        See :doc:`../user_guide/data/data_args` for detail arg setting.
        Note:
            Lower bound of the interval is also called k-core filtering, which means this method
            will filter loops until all the users and items has at least k interactions.
        """
        if self.uid_field is None or self.iid_field is None:
            return

        user_inter_num_interval = self._parse_intervals_str(
            self.config["user_inter_num_interval"]
        )
        item_inter_num_interval = self._parse_intervals_str(
            self.config["item_inter_num_interval"]
        )

        if user_inter_num_interval is None and item_inter_num_interval is None:
            return

        user_inter_num = (
            Counter(self.inter_feat[self.uid_field].values)
            if user_inter_num_interval
            else Counter()
        )
        item_inter_num = (
            Counter(self.inter_feat[self.iid_field].values)
            if item_inter_num_interval
            else Counter()
        )

        n_rm_users = 0
        n_rm_items = 0
        n_rm_inters = 0

        while True:
            ban_users = self._get_illegal_ids_by_inter_num(
                field=self.uid_field,
                feat=self.user_feat,
                inter_num=user_inter_num,
                inter_interval=user_inter_num_interval,
            )
            ban_items = self._get_illegal_ids_by_inter_num(
                field=self.iid_field,
                feat=self.item_feat,
                inter_num=item_inter_num,
                inter_interval=item_inter_num_interval,
            )

            n_rm_users += len(ban_users)
            n_rm_items += len(ban_items)

            if len(ban_users) == 0 and len(ban_items) == 0:
                break

            if self.user_feat is not None:
                dropped_user = self.user_feat[self.uid_field].isin(ban_users)
                self.user_feat.drop(self.user_feat.index[dropped_user], inplace=True)

            if self.item_feat is not None:
                dropped_item = self.item_feat[self.iid_field].isin(ban_items)
                self.item_feat.drop(self.item_feat.index[dropped_item], inplace=True)

            dropped_inter = pd.Series(False, index=self.inter_feat.index)
            user_inter = self.inter_feat[self.uid_field]
            item_inter = self.inter_feat[self.iid_field]
            dropped_inter |= user_inter.isin(ban_users)
            dropped_inter |= item_inter.isin(ban_items)

            user_inter_num -= Counter(user_inter[dropped_inter].values)
            item_inter_num -= Counter(item_inter[dropped_inter].values)

            dropped_index = self.inter_feat.index[dropped_inter]
            n_rm_inters += len(dropped_index)
            self.inter_feat.drop(dropped_index, inplace=True)

        print(
            f"{n_rm_users} users, {n_rm_items} items, {n_rm_inters} interactions have been removed"
            " due to the filter.",
            flush=True,
        )

    def _filter_by_field_value(self):
        """Filter features according to its values."""

        val_intervals = (
            {} if self.config["val_interval"] is None else self.config["val_interval"]
        )

        for field, interval in val_intervals.items():
            if field not in self.field2type:
                raise ValueError(f"Field [{field}] not defined in dataset.")

            if self.field2type[field] in {FeatureType.FLOAT, FeatureType.FLOAT_SEQ}:
                field_val_interval = self._parse_intervals_str(interval)
                for feat in self.field2feats(field):
                    feat.drop(
                        feat.index[
                            ~self._within_intervals(
                                feat[field].values, field_val_interval
                            )
                        ],
                        inplace=True,
                    )
            else:  # token-like field
                for feat in self.field2feats(field):
                    feat.drop(feat.index[~feat[field].isin(interval)], inplace=True)

    def _filter_inter_by_user_or_item(self):
        """Remove interaction in inter_feat which user or item is not in user_feat or item_feat."""
        if self.config["filter_inter_by_user_or_item"] is not True:
            return

        remained_inter = pd.Series(True, index=self.inter_feat.index)

        if self.user_feat is not None:
            remained_uids = self.user_feat[self.uid_field].values
            remained_inter &= self.inter_feat[self.uid_field].isin(remained_uids)

        if self.item_feat is not None:
            remained_iids = self.item_feat[self.iid_field].values
            remained_inter &= self.inter_feat[self.iid_field].isin(remained_iids)

        self.inter_feat.drop(self.inter_feat.index[~remained_inter], inplace=True)

    def _get_illegal_ids_by_inter_num(
        self, field, feat, inter_num, inter_interval=None
    ):
        """Given inter feat, return illegal ids, whose inter num out of [min_num, max_num]
        Args:
            field (str): field name of user_id or item_id.
            feat (pandas.DataFrame): interaction feature.
            inter_num (Counter): interaction number counter.
            inter_interval (list, optional): the allowed interval(s) of the number of interactions.
                                              Defaults to ``None``.
        Returns:
            set: illegal ids, whose inter num out of inter_intervals.
        """
        if inter_interval is not None:
            if len(inter_interval) > 1:
                print(
                    f"More than one interval of interaction number are given!",
                    flush=True,
                )

        ids = {
            id_
            for id_ in inter_num
            if not self._within_intervals(inter_num[id_], inter_interval)
        }

        if feat is not None:
            min_num = inter_interval[0][1] if inter_interval else -1
            for id_ in feat[field].values:
                if inter_num[id_] < min_num:
                    ids.add(id_)
        return ids

    def _parse_intervals_str(self, intervals_str):
        """Given string of intervals, return the list of endpoints tuple, where a tuple corresponds to an interval.
        Args:
            intervals_str (str): the string of intervals, such as "(0,1];[3,4)".
        Returns:
            list of endpoint tuple, such as [('(', 0, 1.0 , ']'), ('[', 3.0, 4.0 , ')')].
        """
        if intervals_str is None:
            return None

        endpoints = []
        for endpoint_pair_str in str(intervals_str).split(";"):
            endpoint_pair_str = endpoint_pair_str.strip()
            left_bracket, right_bracket = endpoint_pair_str[0], endpoint_pair_str[-1]
            endpoint_pair = endpoint_pair_str[1:-1].split(",")
            if not (
                len(endpoint_pair) == 2
                and left_bracket in ["(", "["]
                and right_bracket in [")", "]"]
            ):
                print(f"{endpoint_pair_str} is an illegal interval!", flush=True)
                continue

            left_point, right_point = float(endpoint_pair[0]), float(endpoint_pair[1])
            if left_point > right_point:
                print(f"{endpoint_pair_str} is an illegal interval!", flush=True)

            endpoints.append((left_bracket, left_point, right_point, right_bracket))
        return endpoints

    def _within_intervals(self, num, intervals):
        """return Ture if the num is in the intervals.
        Note:
            return true when the intervals is None.
        """
        result = True
        for i, (left_bracket, left_point, right_point, right_bracket) in enumerate(
            intervals
        ):
            temp_result = num >= left_point if left_bracket == "[" else num > left_point
            temp_result &= (
                num <= right_point if right_bracket == "]" else num < right_point
            )
            result = temp_result if i == 0 else result | temp_result
        return result

    def _del_col(self, feat, field):
        """Delete columns

        Args:
            feat (pandas.DataFrame or Interaction): the feat contains field.
            field (str): field name to be dropped.
        """
        if isinstance(feat, Interaction):
            feat.drop(column=field)
        else:
            feat.drop(columns=field, inplace=True)
        for dct in [
            self.field2id_token,
            self.field2token_id,
            self.field2seqlen,
            self.field2source,
            self.field2type,
        ]:
            if field in dct:
                del dct[field]

    def _reset_index(self):
        """Reset index for all feats in :attr:`feat_name_list`."""
        for feat_name in self.feat_name_list:
            feat = getattr(self, feat_name)
            if feat.empty:
                raise ValueError(
                    "Some feat is empty, please check the filtering settings."
                )
            feat.reset_index(drop=True, inplace=True)

    def _set_alias(self, alias_name, default_value):
        alias = self.config[f"alias_of_{alias_name}"] or []
        alias = np.array(list(filter(None, default_value)) + alias)
        _, idx = np.unique(alias, return_index=True)
        self.alias[alias_name] = alias[np.sort(idx)]

    def _init_alias(self):
        """Set :attr:`alias_of_user_id` and :attr:`alias_of_item_id`. And set :attr:`_rest_fields`."""
        self._set_alias("user_id", [self.uid_field])
        self._set_alias("item_id", [self.iid_field])

        for alias_name_1, alias_1 in self.alias.items():
            for alias_name_2, alias_2 in self.alias.items():
                if alias_name_1 != alias_name_2:
                    intersect = np.intersect1d(alias_1, alias_2, assume_unique=True)
                    if len(intersect) > 0:
                        raise ValueError(
                            f"`alias_of_{alias_name_1}` and `alias_of_{alias_name_2}` "
                            f"should not have the same field {list(intersect)}."
                        )

        self._rest_fields = self.token_like_fields
        for alias_name, alias in self.alias.items():
            isin = np.isin(alias, self._rest_fields, assume_unique=True)
            if isin.all() is False:
                raise ValueError(
                    f"`alias_of_{alias_name}` should not contain "
                    f"non-token-like field {list(alias[~isin])}."
                )
            self._rest_fields = np.setdiff1d(
                self._rest_fields, alias, assume_unique=True
            )

    def _remap_ID_all(self):
        """Remap all token-like fields."""
        for alias in self.alias.values():
            remap_list = self._get_remap_list(alias)
            self._remap(remap_list)

        for field in self._rest_fields:
            remap_list = self._get_remap_list(np.array([field]))
            self._remap(remap_list)

    def _set_label_by_threshold(self):
        """Generate 0/1 labels according to value of features.

        According to ``config['threshold']``, those rows with value lower than threshold will
        be given negative label, while the other will be given positive label.
        See :doc:`../user_guide/data/data_args` for detail arg setting.

        Note:
            Key of ``config['threshold']`` if a field name.
            This field will be dropped after label generation.
        """
        threshold = self.config["threshold"]
        if threshold is None:
            return

        if len(threshold) != 1:
            raise ValueError("Threshold length should be 1.")

        self.set_field_property(
            self.label_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1
        )
        for field, value in threshold.items():
            if field in self.inter_feat:
                self.inter_feat[self.label_field] = (
                    self.inter_feat[field] >= value
                ).astype(int)
            else:
                raise ValueError(f"Field [{field}] not in inter_feat.")
            if field != self.label_field:
                self._del_col(self.inter_feat, field)

    def _get_remap_list(self, field_list):
        """Transfer set of fields in the same remapping space into remap list.
        If ``uid_field`` or ``iid_field`` in ``field_set``,
        field in :attr:`inter_feat` will be remapped firstly,
        then field in :attr:`user_feat` or :attr:`item_feat` will be remapped next, finally others.
        Args:
            field_list (numpy.ndarray): List of fields in the same remapping space.
        Returns:
            list:
            - feat (pandas.DataFrame)
            - field (str)
            - ftype (FeatureType)
            They will be concatenated in order, and remapped together.
        """

        remap_list = []
        for field in field_list:
            ftype = self.field2type[field]
            for feat in self.field2feats(field):
                remap_list.append((feat, field, ftype))
        return remap_list

    def _concat_remaped_tokens(self, remap_list):
        """Given ``remap_list``, concatenate values in order.
        Args:
            remap_list (list): See :meth:`_get_remap_list` for detail.
        Returns:
            tuple: tuple of:
            - tokens after concatenation.
            - split points that can be used to restore the concatenated tokens.
        """
        tokens = []
        for feat, field, ftype in remap_list:
            if ftype == FeatureType.TOKEN:
                tokens.append(feat[field].values)
            elif ftype == FeatureType.TOKEN_SEQ:
                tokens.append(feat[field].agg(np.concatenate))
        split_point = np.cumsum(list(map(len, tokens)))[:-1]
        tokens = np.concatenate(tokens)
        return tokens, split_point

    def _remap(self, remap_list):
        """Remap tokens using :meth:`pandas.factorize`.
        Args:
            remap_list (list): See :meth:`_get_remap_list` for detail.
        """
        if len(remap_list) == 0:
            return
        tokens, split_point = self._concat_remaped_tokens(remap_list)
        new_ids_list, mp = pd.factorize(tokens)
        new_ids_list = np.split(new_ids_list + 1, split_point)
        mp = np.array(["[PAD]"] + list(mp))
        token_id = {t: i for i, t in enumerate(mp)}

        for (feat, field, ftype), new_ids in zip(remap_list, new_ids_list):
            if field not in self.field2id_token:
                self.field2id_token[field] = mp
                self.field2token_id[field] = token_id
            if ftype == FeatureType.TOKEN:
                feat[field] = new_ids
            elif ftype == FeatureType.TOKEN_SEQ:
                split_point = np.cumsum(feat[field].agg(len))[:-1]
                feat[field] = np.split(new_ids, split_point)

    def _check_field(self, *field_names):
        """Given a name of attribute, check if it's exist.
        Args:
            *field_names (str): Fields to be checked.
        """
        for field_name in field_names:
            if getattr(self, field_name, None) is None:
                raise ValueError(f"{field_name} isn't set.")

    def token2id(self, field, tokens):
        """Map external tokens to internal ids.

        Args:
            field (str): Field of external tokens.
            tokens (str, list or numpy.ndarray): External tokens.

        Returns:
            int or numpy.ndarray: The internal ids of external tokens.
        """
        if isinstance(tokens, str):
            if tokens in self.field2token_id[field]:
                return self.field2token_id[field][tokens]
            else:
                raise ValueError(f"token [{tokens}] is not existed in {field}")
        elif isinstance(tokens, (list, np.ndarray)):
            return np.array([self.token2id(field, token) for token in tokens])
        else:
            raise TypeError(f"The type of tokens [{tokens}] is not supported")

    def id2token(self, field, ids):
        """Map internal ids to external tokens.

        Args:
            field (str): Field of internal ids.
            ids (int, list, numpy.ndarray or torch.Tensor): Internal ids.

        Returns:
            str or numpy.ndarray: The external tokens of internal ids.
        """
        try:
            return self.field2id_token[field][ids]
        except IndexError:
            if isinstance(ids, list):
                raise ValueError(f"[{ids}] is not a one-dimensional list.")
            else:
                raise ValueError(f"[{ids}] is not a valid ids.")

    def field2feats(self, field):
        if field not in self.field2source:
            raise ValueError(f"Field [{field}] not defined in dataset.")
        if field == self.uid_field:
            feats = [self.inter_feat]
            if self.user_feat is not None:
                feats.append(self.user_feat)
        elif field == self.iid_field:
            feats = [self.inter_feat]
            if self.item_feat is not None:
                feats.append(self.item_feat)
        else:
            source = self.field2source[field]
            if not isinstance(source, str):
                source = source.value
            feats = [getattr(self, f"{source}_feat")]
        return feats

    def num(self, field):
        """Given ``field``, for token-like fields, return the number of different tokens after remapping,
        for float-like fields, return ``1``.
        Args:
            field (str): field name to get token number.
        Returns:
            int: The number of different tokens (``1`` if ``field`` is a float-like field).
        """
        if field not in self.field2type:
            raise ValueError(f"Field [{field}] not defined in dataset.")

        if (
            self.field2type[field] in {FeatureType.FLOAT, FeatureType.FLOAT_SEQ}
            and field in self.config["numerical_features"]
        ):
            return self.field2bucketnum[field]
        elif self.field2type[field] not in {FeatureType.TOKEN, FeatureType.TOKEN_SEQ}:
            return self.field2seqlen[field]
        else:
            return len(self.field2id_token[field])

    def fields(self, ftype=None, source=None):
        """Given type and source of features, return all the field name of this type and source.
        If ``ftype == None``, the type of returned fields is not restricted.
        If ``source == None``, the source of returned fields is not restricted.
        Args:
            ftype (FeatureType, optional): Type of features. Defaults to ``None``.
            source (FeatureSource, optional): Source of features. Defaults to ``None``.
        Returns:
            list: List of field names.
        """
        ftype = set(ftype) if ftype is not None else set(FeatureType)
        source = set(source) if source is not None else set(FeatureSource)
        ret = []
        for field in self.field2type:
            tp = self.field2type[field]
            src = self.field2source[field]
            if tp in ftype and src in source:
                ret.append(field)
        return ret

    def join(self, df):
        """Given interaction feature, join user/item feature into it.
        Args:
            df (Interaction): Interaction feature to be joint.
        Returns:
            Interaction: Interaction feature after joining operation.
        """
        if self.user_feat is not None and self.uid_field in df:
            df.update(self.user_feat[df[self.uid_field]])
        if self.item_feat is not None and self.iid_field in df:
            df.update(self.item_feat[df[self.iid_field]])
        return df

    def __getitem__(self, index, join=True):
        df = self.inter_feat[index]
        return self.join(df) if join else df

    def set_field_property(self, field, field_type, field_source, field_seqlen):
        """Set a new field's properties.

        Args:
            field (str): Name of the new field.
            field_type (FeatureType): Type of the new field.
            field_source (FeatureSource): Source of the new field.
            field_seqlen (int): max length of the sequence in ``field``.
                ``1`` if ``field``'s type is not sequence-like.
        """
        self.field2type[field] = field_type
        self.field2source[field] = field_source
        self.field2seqlen[field] = field_seqlen

    @property
    def token_like_fields(self):
        """Get fields of type :obj:`~recbole.utils.enum_type.FeatureType.TOKEN` and
        :obj:`~recbole.utils.enum_type.FeatureType.TOKEN_SEQ`.
        Returns:
            list: List of field names.
        """
        return self.fields(ftype=[FeatureType.TOKEN, FeatureType.TOKEN_SEQ])

    @property
    def user_num(self):
        """Get the number of different tokens of ``self.uid_field``.
        Returns:
            int: Number of different tokens of ``self.uid_field``.
        """
        self._check_field("uid_field")
        return self.num(self.uid_field)

    @property
    def item_num(self):
        """Get the number of different tokens of ``self.iid_field``.
        Returns:
            int: Number of different tokens of ``self.iid_field``.
        """
        self._check_field("iid_field")
        return self.num(self.iid_field)

    @property
    def inter_num(self):
        """Get the number of interaction records.
        Returns:
            int: Number of interaction records.
        """
        return len(self.inter_feat)

    @property
    def avg_actions_of_users(self):
        """Get the average number of users' interaction records.
        Returns:
            numpy.float64: Average number of users' interaction records.
        """
        if isinstance(self.inter_feat, pd.DataFrame):
            return np.mean(self.inter_feat.groupby(self.uid_field).size())
        else:
            return np.mean(
                list(Counter(self.inter_feat[self.uid_field].numpy()).values())
            )

    @property
    def avg_actions_of_items(self):
        """Get the average number of items' interaction records.
        Returns:
            numpy.float64: Average number of items' interaction records.
        """
        if isinstance(self.inter_feat, pd.DataFrame):
            return np.mean(self.inter_feat.groupby(self.iid_field).size())
        else:
            return np.mean(
                list(Counter(self.inter_feat[self.iid_field].numpy()).values())
            )

    @property
    def sparsity(self):
        """Get the sparsity of this dataset.
        Returns:
            float: Sparsity of this dataset.
        """
        return 1 - self.inter_num / self.user_num / self.item_num
