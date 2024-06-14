import copy
import numpy as np
import yaml
import importlib

from collections import Counter
from yaml.loader import SafeLoader
from data_augmentation.utils import operator_dict


class AbstractOnlineAugPipeline:
    def __init__(self, dict_path, train_instances):
        self._dict_path = dict_path
        self._instance = train_instances
        self._item_counter = self._item_pop(self._instance)
        self._items = list(self._item_counter.keys())
        self._init()

    def _init(self):
        pass

    def _get_config(self):
        with open(self._dict_path) as f:
            return yaml.load(f, Loader=SafeLoader)

    def _item_pop(self, instances):
        all_items = []
        for seq in instances:
            all_items += seq
        item_pop_counter = Counter(all_items)
        return item_pop_counter

    def __call__(self, seqs, lengths):
        raise NotImplementedError("Pipeline is not implemented in the abstract class.")

    def get_stat(self):
        raise NotImplementedError("Statistic is not implemented in the abstract class.")


class DefaultPipeline(AbstractOnlineAugPipeline):
    def __init__(self, dict_path, train_instances):
        super(DefaultPipeline, self).__init__(dict_path, train_instances)

        print("Initializing default pipeline.")

        self._config = self._get_config()
        self._aug = dict()
        self._aug_configs = dict()
        self._aug_stat = dict()
        self._candidates = ["insert", "replace", "mask", "delete", "subset-split"]
        self._skip_prob = self._config["skip_prob"]
        self._lb = self._config["lower_bound"]
        self._ub = self._config["upper_bound"]
        self._binomial_prob = 0.5

        # adapter
        self._adapter_config_template = {
            "start_pos": 0,
            "end_pos": -1,
            # Insert
            "insert_nums": self._config["insert_items"],
            "percent_no_augment": 0,
            "insert_ratio": 0,
            "insert_n_times": self._config["insert_times"],
            # Replace
            "replace_nums": self._config["replace_items"],
            "replace_ratio": 0,
            "replace_n_times": self._config["replace_times"],
            # Mask
            "mask_nums": self._config["mask_items"],
            "mask_ratio": 0,
            "mask_value": 0,
            "mask_n_times": self._config["mask_times"],
            # Delete
            "delete_nums": self._config["delete_items"],
            "delete_ratio": 0,
            "delete_n_times": self._config["delete_times"],
            # Subset
            "subset_split_n_times": self._config["subset_times"],
            "dropout_prob": self._config["dropout_prob"],
            "pop_counter": self._item_counter,
            "mb_model_name": self._config["mb_model_name"],
            "tempdir": "tmpdir",
            "task_name": "online",
            # Note: don't support time-interval-aware methods so far
        }

        # Load config...
        aug_init_config = {
            "pos": self._config["pos_sample_method"],
            "select": self._config["item_sample_method"],
        }
        aug_init_config.update(self._adapter_config_template)

        for operation in self._candidates:
            # Init config
            operation_config = copy.deepcopy(self._adapter_config_template)
            operation_config["operation_type"] = operation
            self._aug_configs[operation] = operation_config

            # Init operator
            augment_operator = operator_dict[operation](self._items)
            augment_operator.init(
                instances=self._instance, timestamps=None, **aug_init_config
            )
            self._aug[operation] = augment_operator

            # Init counter
            self._aug_stat[operation] = 0

        self._aug_stat["skip"] = 0

    def __call__(self, seqs, lengths):
        aug_seqs = []
        aug_lens = []

        for inst, inst_len in zip(seqs, lengths):
            # trim
            trim_seq = inst[:inst_len]

            if np.random.binomial(1, self._skip_prob, 1)[0] == 1:
                aug_seqs += [trim_seq]
                aug_lens.append(len(trim_seq))
                self._aug_stat["skip"] = self._aug_stat["skip"] + 1
                continue

            operation = "invalid"
            if inst_len <= self._lb:
                operation = "insert"
            else:
                if inst_len <= self._ub:
                    candidates = ["replace", "mask"]
                else:
                    candidates = ["delete", "subset-split"]
                choice = np.random.binomial(1, self._binomial_prob, 1)[0]
                operation = candidates[choice]

            self._aug_stat[operation] = self._aug_stat[operation] + 1
            new_insts, _ = self._aug[operation].forward(
                trim_seq, None, **self._aug_configs[operation]
            )
            aug_seqs += new_insts
            for each_aug_seq in new_insts:
                aug_lens.append(len(each_aug_seq))

        return aug_seqs, aug_lens

    def get_stat(self):
        return self._aug_stat


def get_pipeline(dict_path, train_instances):
    with open(dict_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

        module_path = "data_augmentation.utils.pipeline"
        if importlib.util.find_spec(module_path):
            module = importlib.import_module(module_path)
            pipeline = getattr(module, config["pipeline_name"])(
                dict_path, train_instances
            )
        else:
            raise ValueError(f"Can't find the package")

        if pipeline is None:
            raise ValueError(
                f"Invalid argument 'pipeline_name'[{config['pipeline_name']}] in pipeline settings"
            )

        return pipeline
