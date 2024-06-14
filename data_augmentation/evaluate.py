import copy

import click
import torch
import yaml
import shutil
import os
import xlwt
import pickle
import itertools
from typing import Any

from tabulate import tabulate


from data_augmentation.utils.parallel_utils import dataset_preprocess, ParallelDA

from recbole.data.utils import get_dataloader

from recbole.utils import init_seed, get_model, get_trainer
from recbole.data import create_dataset
from recbole.config import Config


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def add_padding_line(table, row=None, content=""):
    if row:
        table.append(row + [content for i in range(len(table[0]) - len(row))])
    else:
        table.append([content for i in range(len(table[0]))])


def eval_fn(checkpoint, config: Config, eval_args: EasyDict):
    init_seed(seed=eval_args.seed, reproducibility=True)

    # dataset filtering
    dataset = create_dataset(config)

    # dataset splitting
    train_dataset, valid_dataset, test_dataset = dataset.build()
    train_data = get_dataloader(config, "train")(
        config, train_dataset, None, shuffle=True
    )
    test_data = get_dataloader(config, "test")(
        config, test_dataset, None, shuffle=False
    )

    # model loading and initialization
    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    return trainer.evaluate(test_data, load_best_model=False, show_progress=False)


def eval_model(model_type, model_ckpt_path, eval_args: EasyDict):
    print(f"evaluating {model_ckpt_path}")

    saved_ckpt = torch.load(model_ckpt_path)
    eval_task_list = eval_args.eval_task_list

    # align
    num_items = saved_ckpt["state_dict"]["item_embedding.weight"].shape[0]

    config_dict = {
        "data_path": eval_args.temp_dir,
        "gpu_id": 0,
        "benchmark_filename": ["train", "valid", "test"],
        "load_col": {"inter": ["session_id", "item_id_list", "item_id"]},
        "alias_of_item_id": ["item_id_list"],
        "align_num_items": num_items,
        "partitions": "./partitions/solution-1",  # Pop study (hard-coded)
        "partition_dataset": eval_args.dataset,
    }

    test_result = EasyDict()

    if eval_task_list is not None:
        for task_name, task_config_fp in eval_task_list:
            with open(task_config_fp) as f:
                task_config = yaml.load(f, Loader=yaml.SafeLoader)

            test_result[task_name] = EasyDict()

            for subtask_name, subtask_dict in task_config.items():
                config_dict_modified = copy.deepcopy(config_dict)
                config_dict_modified.update(subtask_dict)
                eval_config = Config(
                    model=model_type,
                    dataset=eval_args.task,
                    config_file_list=eval_args.config_file_list,
                    config_dict=config_dict_modified,
                )
                test_result[task_name][subtask_name] = eval_fn(
                    saved_ckpt, eval_config, eval_args
                )
    else:
        config = Config(
            model=model_type,
            dataset=eval_args.task,
            config_file_list=eval_args.config_file_list,
            config_dict=config_dict,
        )
        test_result["no_eval_task"] = eval_fn(saved_ckpt, config, eval_args)

    return test_result


def collect_from_cache(args, cache):
    metrics = ["hit", "mrr", "ndcg"]
    topk = ["@1", "@3", "@5", "@10", "@20", "@50"]
    heads = ["model"]

    for metric, k in itertools.product(metrics, topk):
        heads.append(metric + k)

    table = [heads]

    if not args.with_partitions:
        for identifier, test_result in cache.items():
            line = [identifier]
            for metric in heads[1:]:
                line.append(test_result["no_eval_task"][metric])
            table.append(line)
        print(tabulate(table, headers="firstrow"))
        print("\n")
    else:
        # Export to xlsx by default
        workbook = xlwt.Workbook()
        workbook_name = f"export-pop-study-cr-{args.dataset}.xls"
        table = [[args.dataset] + heads[1:]]
        add_padding_line(table)
        for identifier, all_eval_task in cache.items():
            add_padding_line(table, [identifier])
            for sub_eval_task, eval_result in all_eval_task.items():
                add_padding_line(table, [sub_eval_task])
                for subkey, val in eval_result.items():
                    line = [subkey]
                    for m in heads[1:]:
                        line.append(val[m])
                    table.append(line)

            add_padding_line(table, [])

        sheet = workbook.add_sheet(args.dataset)
        x_dims = len(table)
        y_dims = len(table[0])
        for y in range(y_dims):
            for x in range(x_dims):
                sheet.write(x, y, table[x][y])

        workbook.save(workbook_name)


@click.command()
@click.option(
    "-m",
    "--model_path",
    type=click.Path(file_okay=True),
    default=None,
)
@click.option(
    "-c", "--config", type=click.Path(file_okay=True), default="demo-config.yaml"
)
@click.option(
    "-d",
    "--dataset",
    type=click.Choice(["Amazon_Beauty", "Amazon_Sports", "yelp", "ml-1m"]),
    default="Amazon_Beauty",
)
@click.option(
    "-p", "--dataset_path", type=click.Path(file_okay=False), default="../dataset"
)
@click.option("-b", "--base_aug", type=bool, default=True)
@click.option("-s", "--seed", type=int, default=42)
@click.option("-w", "--with_partitions", type=bool, default=False)
@click.option("-l", "--load_cache", type=click.Path(file_okay=True), default="cache-Amazon_Beauty")
@click.option("-p", "--pop_config", type=click.Path(file_okay=True), default=None)  # "reproductivity/eval_task/pop_ml-1m.yaml"
def cli(**kwargs):
    args = EasyDict(**kwargs)

    if args.load_cache is not None:
        import pickle
        with open(args.load_cache, "rb") as f:
            cache = pickle.load(f)
            print(cache)
            collect_from_cache(args, cache=cache)
            return

    # [("seq_len", "reproductivity/eval_task/seq_len.yaml")] #, [("pop", "reproductivity/eval_task/pop.yaml")]
    args.eval_task_list = (
        [("pop", args.pop_config)] if args.with_partitions else None
    )

    model_list = []
    for model_fp in os.listdir(args.model_path):
        if "SASRec" in model_fp:
            model_type = "SASRec"
        elif "CoSeRec" in model_fp:
            model_type = "CoSeRec"
        elif "ICLRec" in model_fp:
            model_type = "ICLRec"
        elif "CL4SRec" in model_fp:
            model_type = "CL4SRec"
        else:
            raise NotImplemented

        cur_path = os.path.join(args.model_path, model_fp)
        assert len(os.listdir(cur_path)) == 1
        model_ckpt_filepath = os.listdir(cur_path)[0]

        task = "_".join(os.path.basename(cur_path).split("_")[1:-1])

        rcd = (model_type, os.path.join(cur_path, model_ckpt_filepath), task)
        model_list.append(rcd)
        print(rcd)

    temp_dir = f"tmpdir_{args.dataset}"

    file_lists = dataset_preprocess(
        args.dataset,
        args.dataset_path,
        1,
        temp_dir,
        f"configs/config_transform_d_{args.dataset}.yaml",
    )

    da_module = ParallelDA(args.dataset, temp_dir)

    task_list = da_module.generate_all(
        args.dataset,
        file_lists,
        "recbole-slide-window" if args.base_aug else None,
        config_file=args.config,
        on_the_fly=False,
    )

    # hard coded
    task_list = [
        "baseline_recbole-slide-window",
        "O_subset-split",
        "OP_crop_uniform",
        "OP_delete_uniform",
        "OP_mask_uniform",
        "OP_reorder_uniform",
        "OPI_insert_uniform_random",
        "OPI_replace_uniform_random",
    ]
    hard_code_task_list = ["ICLRec", "CoSeRec", "CL4SRec", "O_cl4srec-crop"]
    task_list += hard_code_task_list

    task_list = ["CoSeRec"]

    assert len(model_list) == len(task_list)

    model_configs = dict()
    with open(args.config) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
        for model in params["models"]:
            model_config_list = []
            config_root = "aug_base" if args.base_aug is not None else "raw"
            model_config_filename = params["model_config"][model]
            model_config_list.append(
                os.path.join(
                    params["config_path"],
                    config_root,
                    args.dataset,
                    model_config_filename,
                )
            )
            model_config_list.append(
                os.path.join(
                    params["config_path"],
                    config_root,
                    args.dataset,
                    "config_t_train.yaml",
                )
            )
            model_config_list.append(f"configs/config_d_{args.dataset}.yaml")

            model_configs[model] = model_config_list
            print(f"{model} configs: {model_config_list}")

    results = EasyDict()

    for model_type, model_ckpt_path, task in model_list:
        eval_args = EasyDict()
        eval_args.seed = args.seed
        eval_args.dataset = args.dataset
        eval_args.temp_dir = temp_dir
        eval_args.eval_task_list = args.eval_task_list
        if model_type in hard_code_task_list:
            eval_args.task = "baseline_recbole-slide-window"
        elif task in ["O_cl4srec-crop"]:
            eval_args.task = "baseline_recbole-slide-window"
        else:
            eval_args.task = task
        eval_args.config_file_list = model_configs[model_type]
        test_result = eval_model(model_type, model_ckpt_path, eval_args)

        identifier = (
            f"{task}_{model_type}"  # os.path.basename(model_ckpt_path).split(".")[0]
        )
        results[identifier] = test_result

    import pickle

    with open(f"cache-{args.dataset}", "wb") as f:
        pickle.dump(results, f)

    collect_from_cache(args, cache=results)

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    cli()
