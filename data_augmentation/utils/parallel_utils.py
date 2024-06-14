import copy
import os
import yaml
import pickle
import time
import shutil

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import BoundedSemaphore

from data_augmentation.utils.data_transform import *
from data_augmentation.utils.trim_dataset import TrimDataset
from data_augmentation.utils.pipeline import get_pipeline
from yaml.loader import SafeLoader
from collections import Counter
from data_augmentation.utils import (
    operator_dict,
    ti_operator_dict,
    need_unique_position_sampling,
    unique_position_sampling_mapping,
    unique_operator_dict,
)

from recbole.config import Config
from recbole.data import (
    create_dataset,
)
from recbole.data.utils import get_aug_dataloader, get_dataloader
from recbole.utils import (
    get_model,
    get_trainer,
    init_seed,
)


tmp_id = "_tmp"


def fetch_og_path(dataset_path, dataset_name):
    return os.path.join(dataset_path, dataset_name)


def fetch_tmp_path(dataset_path, dataset_name, create_dir=False):
    tmp_path = os.path.join(dataset_path, dataset_name + tmp_id)
    if create_dir:
        os.makedirs(tmp_path, exist_ok=True)
    return tmp_path


def fetch_tmp_dataset(dataset_name):
    return dataset_name + tmp_id


def init_dataset(transform_config_path, dataset_path, dataset_name):
    config = None
    with open(transform_config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)
    return TrimDataset(config, dataset_name, dataset_path)


def load_train_session(dataset_path, dataset_name, cold_start_ratio):
    print("Loading existing sessions.", flush=True)
    start_time = time.time()
    sessions = dict()
    sessions_time = dict()

    session_file_path = os.path.join(dataset_path, f"{dataset_name}.train.inter")
    session_time_path = os.path.join(dataset_path, f"{dataset_name}.train.time")

    session_id = 0
    with open(session_file_path, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            item_id_list = " ".join([line.split("\t")[1], line.split("\t")[2]]).split()
            sessions[str(session_id)] = item_id_list
            session_id += 1

    session_id = 0
    with open(session_time_path, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            item_id_list = " ".join([line.split("\t")[1], line.split("\t")[2]]).split()
            sessions_time[str(session_id)] = item_id_list
            session_id += 1

    if len(sessions) != len(sessions_time):
        raise ValueError("mismatched sequence size.")

    if cold_start_ratio != 1:
        sessions, sessions_time = extract_cold_start_trivial(
            sessions, sessions_time, cold_start_ratio
        )

    end_time = time.time()
    print(
        f"Session load finished, elapsed: {round(end_time-start_time, 4)} s",
        flush=True,
    )
    return sessions, sessions_time


def import_existing_dataset(
    dataset_path,
    dataset_name,
    cold_start_ratio,
    output_path,
):
    tmp_path = fetch_tmp_path(output_path, dataset_name, True)
    tmp_dataset = fetch_tmp_dataset(dataset_name)

    train_file_path = os.path.join(tmp_path, f"{tmp_dataset}.train.inter")
    valid_file_path = os.path.join(tmp_path, f"{tmp_dataset}.valid.inter")
    test_file_path = os.path.join(tmp_path, f"{tmp_dataset}.test.inter")
    file_list = [train_file_path, valid_file_path, test_file_path]

    dataset = os.path.join(dataset_path, dataset_name)

    print(f"Dataset '{dataset_name}' preprocessing started.")
    start_time = time.time()

    train, train_time = load_train_session(dataset, dataset_name, cold_start_ratio)

    # Export tmp files
    export_file(train_file_path, train, train_time)

    # Copy the validation and test data
    for token in ["valid", "test"]:
        session_file_path = os.path.join(dataset, f"{dataset_name}.{token}.inter")
        session_time_path = os.path.join(dataset, f"{dataset_name}.{token}.time")
        shutil.copyfile(
            session_file_path,
            os.path.join(tmp_path, f"{tmp_dataset}.{token}.inter"),
        )
        shutil.copyfile(
            session_time_path,
            os.path.join(tmp_path, f"{tmp_dataset}.{token}.time"),
        )

    end_time = time.time()
    print(
        f"Dataset preprocessing finished, elapsed: {round(end_time-start_time, 4)} s.",
        flush=True,
    )

    return file_list


def dataset_preprocess(
    dataset_name,
    dataset_path,
    cold_start_ratio,
    output_path,
    transform_config_path,
):
    tmp_path = fetch_tmp_path(output_path, dataset_name, True)
    tmp_dataset = fetch_tmp_dataset(dataset_name)

    train_file_path = os.path.join(tmp_path, f"{tmp_dataset}.train.inter")
    valid_file_path = os.path.join(tmp_path, f"{tmp_dataset}.valid.inter")
    test_file_path = os.path.join(tmp_path, f"{tmp_dataset}.test.inter")
    file_list = [train_file_path, valid_file_path, test_file_path]

    print(f"Dataset '{dataset_name}' preprocessing started.")
    start_time = time.time()

    dataset = init_dataset(transform_config_path, dataset_path, dataset_name)
    sessions, sessions_time = generate_session(dataset)

    train, valid, test, train_time, valid_time, test_time = extract_train_valid_test(
        sessions, sessions_time
    )

    if cold_start_ratio != 1:
        train, train_time = extract_cold_start_trivial(
            train, train_time, cold_start_ratio
        )

    # Export tmp files
    export_file(train_file_path, train, train_time)
    export_file(valid_file_path, valid, valid_time)
    export_file(test_file_path, test, test_time)

    end_time = time.time()
    print(
        f"Dataset preprocessing finished, elapsed: {round(end_time-start_time, 4)} s.",
        flush=True,
    )

    return file_list


class ParallelDA(object):
    def __init__(self, dataset_name, dataset_path):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.tasks = set()
        self.semaphore = BoundedSemaphore(1)
        self.executor_pool = None
        self.indicator = None
        self.out_semaphore = BoundedSemaphore(1)
        self.max_worker = 4

        self.train_instances = None
        self.item_counter = None
        self.online_config = None

        self.submit_set = set()

    def generate_all(
        self, dataset_name, inter_file_list, aug_base, config_file, on_the_fly
    ):
        for each_inter_file in inter_file_list:
            print(
                f"Data augmentation for file '{each_inter_file}' started.", flush=True
            )
            self.executor_pool = ThreadPoolExecutor(self.max_worker)
            tokens = each_inter_file.split(".")
            if tokens[-2] in ["train"]:
                self.indicator = tokens[-2]
            elif tokens[-2] in ["valid", "test"]:

                print(
                    f"Data augmentation for file '{each_inter_file}' skipped.",
                    flush=True,
                )
                continue
            elif tokens[-2] in ["cold_start"]:
                self.indicator = ""
                continue
            else:
                raise ValueError("")
            instances = self.read_inter(each_inter_file)
            timestamps = self.read_time(each_inter_file.replace(".inter", ".time"))

            # Note: call once!
            self.train_instances = instances
            self.item_counter = self.item_pop(instances)

            # Check aug_base
            # Disable aug_base when on_the_fly is on
            if aug_base is not None and not on_the_fly:
                task_name = f"baseline_{aug_base}"
                print(f"Data augmenting for aug_base: '{task_name}'")

                params = None
                with open(config_file) as f:
                    params = yaml.load(f, Loader=SafeLoader)
                params["dataset"] = dataset_name
                item_counter = self.item_pop(instances)
                params["pop_counter"] = item_counter
                items = list(item_counter.keys())
                augment_operator = operator_dict[aug_base](items)

                # Note: Only support unique DA methods
                config = params.copy()
                config["operation_type"] = aug_base
                start_time = time.time()
                aug_seqs = []  # copy.deepcopy(instances)
                aug_ts = []  # copy.deepcopy(timestamps)
                for seq, ts in zip(instances, timestamps):
                    augmented_seq, augmented_ts = augment_operator.forward(
                        seq, ts, **config
                    )
                    aug_seqs += augmented_seq
                    aug_ts += augmented_ts

                instances = aug_seqs
                timestamps = aug_ts

                del aug_seqs, aug_ts

                end_time = time.time()
                print(
                    f"Data augmentation for aug_base '{task_name}' finished, elapsed: {round(end_time - start_time, 4)} s.",
                    flush=True,
                )

                self.save_data(instances, timestamps, task_name)
                self.submit_set.add(task_name)
            else:
                # Generate the baseline test
                self.save_data(instances, timestamps, "baseline")
                self.submit_set.add("baseline")

            # if on_the_fly is on, skip the offline augmentation
            if on_the_fly:
                print(
                    f"Performing online data augmentation, offline augmentation for file {each_inter_file} skipped."
                )
                self.save_data(instances, timestamps, "online")
                self.submit_set.add("online")

                with open(config_file) as f:
                    params = yaml.load(f, Loader=SafeLoader)
                    self.online_config = params["online_dict_path"]
                continue

            self.data_augment(
                dataset_name, instances, timestamps, config_file=config_file
            )
            self.executor_pool.shutdown(wait=True)
            self.executor_pool = None

            print(
                f"Data augmentation for file '{each_inter_file}' finished.", flush=True
            )

        diff = self.submit_set - self.tasks
        if len(diff) != 0:
            print(f"Missing tasks: {diff}")
        tasks = list(self.tasks)
        for each_inter_file in inter_file_list:
            tokens = each_inter_file.split(".")
            if tokens[-2] not in ["valid", "test"]:
                continue
            else:
                for task_name in tasks:
                    dst_dir = os.path.join(self.dataset_path, task_name)
                    shutil.copyfile(
                        each_inter_file,
                        os.path.join(dst_dir, task_name + "." + tokens[-2] + ".inter"),
                    )

        return tasks

    def item_pop(self, instances):
        all_items = []
        for seq in instances:
            all_items += seq
        item_pop_counter = Counter(all_items)
        return item_pop_counter

    @staticmethod
    def read_inter(path):
        instances = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                item_id_list = " ".join(
                    [line.split("\t")[1], line.split("\t")[2]]
                ).split()
                instances.append(item_id_list)
        return instances

    @staticmethod
    def read_time(path):
        timestamps = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                timestamps_list = " ".join(
                    [line.split("\t")[1], line.split("\t")[2]]
                ).split()
                timestamps_list = [
                    int(each_timestamp) for each_timestamp in timestamps_list
                ]
                timestamps.append(timestamps_list)
        return timestamps

    def save_inter(self, root, flag, sequences, timestamps):
        self.semaphore.acquire()
        self.tasks.add(flag)
        print(f"{len(sequences)} sequences augmented for task '{flag}'")
        self.semaphore.release()
        folder = os.path.join(root, flag)
        os.makedirs(folder, exist_ok=True)
        inter_path = os.path.join(folder, f"{flag}.{self.indicator}.inter")
        session_id = 1
        with open(inter_path, "w") as f:
            f.write("session_id:token\titem_id_list:token_seq\titem_id:token\n")
            for seq in sequences:
                # cvt2str
                seq_s = list(map(str, seq))
                f.write(
                    str(session_id)
                    + "\t"
                    + " ".join(seq_s[:-1])
                    + "\t"
                    + seq_s[-1]
                    + "\n"
                )
                session_id += 1

        # session_id = 1
        # if timestamps is not None:
        #     with open(os.path.join(folder, f"{flag}.{self.indicator}.time"), "w") as f:
        #         f.write("session_id:token\ttimestamp_list:float_seq\ttimestamp:float\n")
        #         for ts in timestamps:
        #             ts = list(map(str, ts))
        #             f.write(
        #                 str(session_id) + "\t" + " ".join(ts[:-1]) + "\t" + ts[-1] + "\n"
        #             )
        #             session_id += 1

    def save_aug_base(self, root, flag, sequences, timestamps):
        self.semaphore.acquire()
        self.tasks.add(flag)
        print(f"{len(sequences)} sequences augmented for task '{flag}'")
        self.semaphore.release()
        folder = os.path.join(root, flag)
        os.makedirs(folder, exist_ok=True)
        inter_path = os.path.join(folder, f"{flag}.{self.indicator}.inter")
        session_id = 1
        with open(inter_path, "w") as f:
            f.write("session_id:token\titem_id_list:token_seq\titem_id:token\n")
            for seq in sequences:
                # cvt2str
                seq_s = list(map(str, seq))
                f.write(
                    str(session_id)
                    + "\t"
                    + " ".join(seq_s[:-1])
                    + "\t"
                    + seq_s[-1]
                    + "\n"
                )
                session_id += 1

        session_id = 1
        if timestamps is not None:
            with open(os.path.join(folder, f"{flag}.{self.indicator}.time"), "w") as f:
                f.write("session_id:token\ttimestamp_list:float_seq\ttimestamp:float\n")
                for ts in timestamps:
                    ts = list(map(str, ts))
                    f.write(
                        str(session_id)
                        + "\t"
                        + " ".join(ts[:-1])
                        + "\t"
                        + ts[-1]
                        + "\n"
                    )
                    session_id += 1

    def save_data(self, sequences, timestamps, task_name, **kwargs):
        self.save_inter(
            root=self.dataset_path,
            flag=task_name,
            sequences=sequences,
            timestamps=timestamps,
        )

    def dispatch(self, task_name, instances, timestamps, augmentor, **kwargs):
        """
        Dispatch the augmentation operation to different specific post-processes.
        """
        try:
            operation_type = kwargs["operation_type"]
            print(f"Data augmentation '{task_name}' started.", flush=True)
            augmentor.init(instances, timestamps, task_name=task_name, **kwargs)
            start_time = time.time()
            if operation_type in ti_operator_dict:
                aug_seqs = copy.deepcopy(instances)
                aug_ts = copy.deepcopy(timestamps)
                for seq, ts in zip(instances, timestamps):
                    augmented_seq, augmented_ts = augmentor.forward(seq, ts, **kwargs)
                    aug_seqs += augmented_seq
                    aug_ts += augmented_ts
                self.save_data(aug_seqs, aug_ts, task_name, **kwargs)
            elif operation_type in operator_dict:
                aug_seqs = copy.deepcopy(instances)
                for seq in instances:
                    augmented_seq, _ = augmentor.forward(seq, None, **kwargs)
                    aug_seqs += augmented_seq
                self.save_data(aug_seqs, None, task_name, **kwargs)
            else:
                raise ValueError(
                    f"Unsupported dispatch implementation for operation type {operation_type}"
                )
        except Exception:
            self.out_semaphore.acquire()
            print(f"{task_name} fails...")
            import traceback

            traceback.print_exc()
            self.out_semaphore.release()
        else:
            end_time = time.time()
            self.out_semaphore.acquire()
            print(
                f"Data augmentation '{task_name}' finished, elapsed: {round(end_time - start_time, 4)} s.",
                flush=True,
            )
            self.out_semaphore.release()

    def data_augment(self, dataset_name, instances, timestamps, **kwargs):
        """
        **kwarg:
        config_file:        (str) the path of configuration for data augmentation.
        dataset:            (str) the dataset name
        """

        config_path = kwargs["config_file"]

        params = None
        with open(config_path) as f:
            params = yaml.load(f, Loader=SafeLoader)

            if params["augment_operation"] is None:
                print("No offline augmentation.", flush=True)
                return

        # Get the popularity counter
        item_counter = self.item_pop(instances)
        params["dataset"] = dataset_name
        params["pop_counter"] = item_counter
        params["tempdir"] = self.dataset_path
        items = list(item_counter.keys())

        for op_type in params["augment_operation"]:
            augment_operator = operator_dict[op_type](items)
            if op_type in unique_operator_dict:
                config = params.copy()
                config["operation_type"] = op_type
                task_name = f"O_{op_type}"
                print(f"Submit data augmentation task '{task_name}'")
                self.submit_set.add(task_name)
                self.executor_pool.submit(
                    self.dispatch,
                    task_name,
                    instances,
                    timestamps,
                    augmentor=copy.deepcopy(augment_operator),
                    **config,
                )
                continue

            position_sampling_methods = (
                unique_position_sampling_mapping[op_type]
                if op_type in need_unique_position_sampling
                else params["pos_sample_method"]
            )

            for each_candidate_method in position_sampling_methods:
                if op_type in params["need_item_sampling"]:
                    for item_sampling in params["item_sample_method"]:
                        config = params.copy()
                        config["pos"] = each_candidate_method
                        config["select"] = item_sampling
                        config["operation_type"] = op_type
                        task_name = (
                            f"OPI_{op_type}_{each_candidate_method}_{item_sampling}"
                        )
                        print(f"Submit data augmentation task '{task_name}'")
                        self.submit_set.add(task_name)
                        self.executor_pool.submit(
                            self.dispatch,
                            task_name,
                            instances,
                            timestamps,
                            augmentor=copy.deepcopy(augment_operator),
                            **config,
                        )
                else:
                    config = params.copy()
                    config["pos"] = each_candidate_method
                    config["operation_type"] = op_type
                    task_name = f"OP_{op_type}_{each_candidate_method}"
                    print(f"Submit data augmentation task '{task_name}'")
                    self.submit_set.add(task_name)
                    self.executor_pool.submit(
                        self.dispatch,
                        task_name,
                        instances,
                        timestamps,
                        augmentor=copy.deepcopy(augment_operator),
                        **config,
                    )


def add_unique_task(config_file, root, aug_base):
    unique_task_list = []
    ablation_unique_task_list = []
    baseline_task = "baseline"
    if aug_base is not None:
        baseline_task = f"baseline_{aug_base}"

    baseline = os.path.join(root, baseline_task)
    with open(config_file) as f:
        params = yaml.load(f, Loader=SafeLoader)
        if params["unique_task"] is None:
            return unique_task_list, ablation_unique_task_list

        for unique_task in params["unique_task"]:
            subtask_list = []
            # check if it needs to run ablation studies exps
            if (
                params["ablation_task_config"] is not None
                and unique_task in params["ablation_task_config"]
            ):
                ablation_list = params["ablation_task_config"][unique_task]
                for item in ablation_list:
                    task_name = f"{unique_task}-{item}"
                    subtask_list.append(task_name)
                    ablation_unique_task_list.append(task_name)
            else:
                subtask_list.append(unique_task)

            for subtask in subtask_list:
                unique_task_list.append(subtask)
                unique_task_path = os.path.join(root, subtask)
                os.makedirs(unique_task_path, exist_ok=True)
                for data_file in os.listdir(baseline):
                    token = data_file.split(".")[-2]
                    shutil.copyfile(
                        os.path.join(baseline, data_file),
                        os.path.join(
                            unique_task_path, subtask + "." + token + ".inter"
                        ),
                    )

    return unique_task_list, ablation_unique_task_list


def run_task(
    task,
    model,
    model_configs,
    gpu_id,
    tmpdir,
    on_the_fly,
    ablation,
    train_instances,
    on_the_fly_dict_path,
    ckpt_path,
):
    # configurations initialization
    config_dict = {
        "data_path": tmpdir,
        "gpu_id": gpu_id,
        "benchmark_filename": ["train", "valid", "test"],
        "USER_ID_FIELD": "session_id",
        "ITEM_ID_FIELD": "item_id",
        "load_col": {"inter": ["session_id", "item_id_list", "item_id"]},
        "alias_of_item_id": ["item_id_list"],
        "checkpoint_dir": os.path.join(ckpt_path, f"cpt_{task}_{model}"),
        "save_dir": "./out",
    }

    # ablation
    if ablation:
        config_dict.update({task.split("-")[1]: True})

    config_file_list = model_configs[model]

    config = Config(
        model=model,
        dataset=task,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )

    start_time = time.time()

    seed = config["seed"]
    init_seed(seed=seed, reproducibility=True)

    # dataset filtering
    dataset = create_dataset(config)

    # dataset splitting
    train_dataset, valid_dataset, test_dataset = dataset.build()
    dataloader = get_aug_dataloader(config, phase="train", on_the_fly=on_the_fly)
    if on_the_fly:
        pipeline = get_pipeline(on_the_fly_dict_path, train_instances)
        train_data = dataloader(
            config, train_dataset, None, shuffle=True, aug_pipeline=pipeline
        )
    else:
        train_data = dataloader(config, train_dataset, None, shuffle=True)
    valid_data = get_dataloader(config, "valid")(
        config, valid_dataset, None, shuffle=False
    )
    test_data = get_dataloader(config, "test")(
        config, test_dataset, None, shuffle=False
    )

    # model loading and initialization
    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model training
    (
        best_valid_score,
        best_valid_result,
        train_epoch_cost,
        valid_epoch_cost,
    ) = trainer.fit(train_data, valid_data, saved=True, show_progress=False)

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=False)

    end_time = time.time()

    return {
        "task": task,
        "gpu": gpu_id,
        "model": config["model"],
        "best_valid_score": best_valid_score,
        "best_valid_result": best_valid_result,
        "train_epoch_cost": train_epoch_cost,
        "valid_epoch_cost": valid_epoch_cost,
        "test_result": test_result,
        "elapsed": end_time - start_time,
        "seqs": len(train_dataset),
        "seed": seed,
        "online_stat": "N/A" if not on_the_fly else train_data.get_online_stat(),
    }


class ParallelRun(object):
    def __init__(
        self,
        tasks,
        unique_tasks,
        ablation_unique_task_list,
        num_gpus,
        num_tasks_per_gpu,
        config_file,
        **kwargs,
    ):
        self.task_queue = tasks
        self.unique_task_queue = unique_tasks
        self.ablation_unique_task_list = ablation_unique_task_list
        self.num_gpus = num_gpus
        self.num_tasks_per_gpu = num_tasks_per_gpu
        self.owned_devices = []
        self.physical_gpu_id_map = {}

        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        for i, device in enumerate(cuda_visible_devices):
            if i >= self.num_gpus:
                break

            physical_gpu_id = device.strip()
            self.owned_devices.append(physical_gpu_id)
            self.physical_gpu_id_map[physical_gpu_id] = i

        assert self.num_gpus == len(self.owned_devices)

        self.model = kwargs["model"]  # baseline model, i.e., SASRec
        self.tmpdir = kwargs["tmpdir"]
        self.output_path = kwargs["output_path"]
        self.on_the_fly = kwargs["on_the_fly"]
        self.train_instances = kwargs["train_instances"]
        self.on_the_fly_dict_path = kwargs["on_the_fly_dict_path"]
        self.partitions = kwargs["partitions"]
        self.partition_dataset = kwargs["partition_dataset"]
        self.has_aug_base = kwargs["has_aug_base"]
        self.dataset_name = kwargs["dataset_name"]
        self.ckpt_path = kwargs["ckpt_path"]

        print(f"partition: {self.partitions}")

        os.makedirs(self.output_path, exist_ok=True)

        self.n_total = self.num_gpus * self.num_tasks_per_gpu
        self.gpu_resources = [
            BoundedSemaphore(num_tasks_per_gpu) for _ in range(num_gpus)
        ]
        self.avails = BoundedSemaphore(self.n_total)

        self.n_tasks = len(self.task_queue)
        self.n_unique_tasks = len(self.unique_task_queue)

        self.global_cnt = 0
        self.global_cnt_semaphore = BoundedSemaphore(1)

        # For reproductivity
        self.model_configs = dict()
        with open(config_file) as f:
            params = yaml.load(f, Loader=SafeLoader)
            for model in params["models"]:
                model_config_list = []
                config_root = "aug_base" if self.has_aug_base else "raw"
                model_config_filename = params["model_config"][model]
                model_config_list.append(
                    os.path.join(
                        params["config_path"],
                        config_root,
                        self.dataset_name,
                        model_config_filename,
                    )
                )
                model_config_list.append(
                    os.path.join(
                        params["config_path"],
                        config_root,
                        self.dataset_name,
                        "config_t_train.yaml",
                    )
                )
                model_config_list.append(f"configs/config_d_{self.dataset_name}.yaml")

                self.model_configs[model] = model_config_list
                print(f"{model} configs: {model_config_list}")

    def callback_fn(self, res):
        info = res.result()
        # 释放GPU资源
        gpu_resource_id = self.physical_gpu_id_map[info["gpu"]]
        self.gpu_resources[gpu_resource_id].release()

        self.global_cnt_semaphore.acquire()
        self.global_cnt += 1
        print(
            f"task {info['task']} on gpu {info['gpu']} completed, elapsed: {round(info['elapsed'], 4)} s, progress: {self.global_cnt}/{self.n_tasks}",
            flush=True,
        )
        self.global_cnt_semaphore.release()

        info["test_result"]["number of instances"] = info["seqs"]
        info["test_result"]["seed"] = info["seed"]
        info["test_result"]["train_epoch_cost"] = info["train_epoch_cost"]
        info["test_result"]["valid_epoch_cost"] = info["valid_epoch_cost"]
        if isinstance(info["online_stat"], str):
            padding = {
                "insert": "N/A",
                "replace": "N/A",
                "mask": "N/A",
                "delete": "N/A",
                "subset-split": "N/A",
                "skip": "N/A",
            }
            info["test_result"].update(padding)
        else:
            info["test_result"].update(info["online_stat"])
        with open(os.path.join(self.output_path, info["task"]), "wb") as f:
            pickle.dump(info["test_result"], f)

        self.avails.release()

    def run(self):
        with ProcessPoolExecutor(max_workers=self.n_total) as executor:
            while True:
                task_queue = self.task_queue

                with open("configs/pop.yaml") as f:
                    task_config = yaml.load(f, Loader=SafeLoader)

                # for subtask_name, subtask_dict in task_config.items():
                for task in task_queue:
                    self.avails.acquire()

                    allocated_gpu_id = -1
                    for gpu_id in range(len(self.gpu_resources)):
                        if self.gpu_resources[gpu_id].acquire(blocking=False):
                            allocated_gpu_id = gpu_id
                            break

                    if allocated_gpu_id == -1:
                        raise ValueError("Unable to allocate available gpu resource.")

                    physical_gpu_id = self.owned_devices[allocated_gpu_id]

                    print(
                        f"Start running '{task}' on gpu {physical_gpu_id}",
                        flush=True,
                    )

                    if task in self.ablation_unique_task_list:
                        model_name = task.split("-")[0]
                    elif task in self.unique_task_queue:
                        model_name = task
                    else:
                        model_name = self.model

                    executor.submit(
                        run_task,
                        task,
                        model_name,  # model
                        self.model_configs,
                        physical_gpu_id,
                        self.tmpdir,
                        True if task == "online" else False,
                        True if task in self.ablation_unique_task_list else False,
                        self.train_instances,
                        self.on_the_fly_dict_path,
                        self.ckpt_path,
                    ).add_done_callback(self.callback_fn)

                # Latch
                executor.shutdown(wait=True)
                print("Done", flush=True)

                return
