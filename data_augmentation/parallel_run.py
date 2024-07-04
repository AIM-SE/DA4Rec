import os
import time
import shutil
import argparse

from data_augmentation.utils.parallel_utils import (
    dataset_preprocess,
    ParallelDA,
    ParallelRun,
    add_unique_task,
)

from recbole.utils import (
    init_seed,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", help="model name", type=str, default="SASRec")
    parser.add_argument("--dataset_path", help="path to directory of datasets", type=str, default="../dataset")
    parser.add_argument("--dataset", help="dataset name", type=str, default="Amazon_Beauty")
    parser.add_argument("--on_the_fly", action="store_true", default=False)
    parser.add_argument("--aug_base", help="base augmentation method", type=str, default=None)
    parser.add_argument("--cold_start_ratio", help="cold-start sampling ratio", type=float, default=1)
    parser.add_argument("--seed", help="random seed", type=int, default=42)

    parser.add_argument("--n_gpus", help="how many gpus", type=int, default=1)
    parser.add_argument("--n_task_per_gpu", help="how many tasks per gpu", type=int, default=1)

    parser.add_argument("--output_path", help="path to save results", type=str, default=None)
    parser.add_argument("--ckpt_path", help="path to save checkpoints", type=str, default=None)
    parser.add_argument("--partitions", type=str, default=None)
    parser.add_argument("--da_config", help="configuration to data augmentation", type=str, default="demo-config.yaml")
    parser.add_argument("--not_clean", help="whether not to delete tmp files", action="store_true", default=False)
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = f"{args.dataset}_{args.cold_start_ratio}"

    temp_dir = f"tmpdir_{args.dataset}_{time.time(): .4f}"
    print(f"Temp: {temp_dir}")

    if args.ckpt_path is None:
        args.ckpt_path = temp_dir
        os.makedirs(args.ckpt_path, exist_ok=True)

    file_lists = dataset_preprocess(
        args.dataset,
        args.dataset_path,
        args.cold_start_ratio,
        temp_dir,
        f"configs/config_transform_d_{args.dataset}.yaml",
    )

    init_seed(seed=args.seed, reproducibility=True)

    da_module = ParallelDA(args.dataset, temp_dir)

    print(f"Start data augmentation for '{args.dataset}'", flush=True)
    task_list = da_module.generate_all(
        args.dataset,
        file_lists,
        args.aug_base,
        config_file=args.da_config,
        on_the_fly=args.on_the_fly,
    )
    unique_task_list, ablation_unique_task_list = add_unique_task(
        args.da_config, temp_dir, args.aug_base
    )
    print(f"Number of data augmentations: {len(task_list) - 1}", flush=True)
    task_list = task_list + unique_task_list
    print(f"Number of total tests: {len(task_list)}")
    print("Following tasks will be tested:")
    for each_task in task_list:
        print(f"\t{each_task}", flush=True)
    print(f"Start benchmarking for {args.model}", flush=True)
    print(f"Number of GPUs: {args.n_gpus}", flush=True)
    print(f"Number of tasks per gpu: {args.n_task_per_gpu}", flush=True)
    print(f"Result output path: '{args.output_path}'", flush=True)

    engine = ParallelRun(
        tasks=task_list,
        unique_tasks=unique_task_list,
        ablation_unique_task_list=ablation_unique_task_list,
        num_gpus=args.n_gpus,
        num_tasks_per_gpu=args.n_task_per_gpu,
        config_file=args.da_config,
        model=args.model,
        tmpdir=temp_dir,
        output_path=args.output_path,
        on_the_fly=args.on_the_fly,
        train_instances=da_module.train_instances,
        on_the_fly_dict_path=da_module.online_config,
        partitions=args.partitions,
        partition_dataset=args.dataset,
        has_aug_base=(args.aug_base is not None),
        dataset_name=args.dataset,
        ckpt_path=args.ckpt_path,
    )
    engine.run()

    if args.not_clean is False and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
