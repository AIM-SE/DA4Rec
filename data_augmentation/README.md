## Quick Start

1. install all the prerequisite packages required by Recbole
2. config PYTHONPATH: export PYTHONPATH=/XXX/PROJECT_ROOT
3. Prepare the dataset to '/dataset'
4. Config data augmentation settings (see demo-config.yaml)

Arguments:
   * --model: baseline model  (default: SASRec)
   * --dataset_path: path to directory of datasets (default: ../dataset)
   * --dataset: dataset name (default: Amazon_Beauty)
   * --cold_start_ratio: cold-start sampling ratio (default: 1)
   * --n_gpus: how many gpus (default: 1)
   * --n_task_per_gpu: how many tasks per gpu (default: 1)
   * --output_path: path to save results (default: None)
   * --ckpt_path: path to save checkpoints (default: None)
   * --da_config: configuration to data augmentation (default: demo-config.yaml)
   * --not_clean: whether not to delete tmp files (default: False)
   * --aug_base: base augmentation method (default: None, "recbole-slide-window" for replica)
   * --seed: random seed (default: 42)
   * --partitions: path to items list partitioned by popularity (default: None) (Note: only used for evaluation)


```python
CUDA_VISIBLE_DEVICES="0,1,2" python parallel_run.py --model=SASRec --dataset=Amazon_Beauty [--aug_base=recbole-slide-window] --n_gpus=3 --n_task_per_gpu=2 --output_path="result-outs" --ckpt_path="ckpt-outs" --da_config="demo-config.yaml"
```

## Repository Structure

```
data_augmentation
├─configs:             configurations
├─reproductivity:      configurations for reproductions
├─scripts:             scripts for reproductions
├─partitions:          item list partitioned by item popularity
├─da_operator:         operation of data augmentation
├─utils:               utilities
├─demo-config.yaml:    demo configuration file
├─da_analyzer.py:      analysis script for .xlsx export
└─parallel_run.py:     entrypoint

```
CUDA_VISIBLE_DEVICES="0,1,2" python parallel_run.py --model=SASRec --dataset=Amazon_Beauty --aug_base=recbole-slide-window --n_gpus=3 --n_task_per_gpu=2 --output_path="result-outs" --ckpt_path="ckpt-outs" --da_config="demo-config.yaml"

export PYTHONPATH=/home/yueqi/peilin/DA4Rec
CUDA_VISIBLE_DEVICES="0,1,2" python data_augmentation/parallel_run.py --model=SASRec --dataset=Amazon_Beauty [--aug_base=recbole-slide-window] --n_gpus=3 --n_task_per_gpu=2 --output_path="result-outs" --ckpt_path="ckpt-outs" --da_config="demo-config.yaml"
