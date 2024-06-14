
SHELL_FOLDER=$(cd "$(dirname "$0")"; pwd)
ROOT_FOLDER=$SHELL_FOLDER/..
OUT_FOLDER=$ROOT_FOLDER/outs

CUDA_VISIBLE_DEVICES="0, 1, 2" python $ROOT_FOLDER/parallel_run.py --model=SASRec --dataset=Amazon_Beauty --n_gpus=3 --n_task_per_gpu=2 --output_path="${OUT_FOLDER}/Beauty-result-outs" --ckpt_path="${OUT_FOLDER}/Beauty-ckpt-outs" --da_config=$ROOT_FOLDER/reproductivity/configs/raw/Amazon_Beauty/aug-config.yaml

CUDA_VISIBLE_DEVICES="0, 1, 2" python $ROOT_FOLDER/parallel_run.py --model=SASRec --dataset=Amazon_Sports --n_gpus=3 --n_task_per_gpu=2 --output_path="${OUT_FOLDER}/Sports-result-outs" --ckpt_path="${OUT_FOLDER}/Sports-ckpt-outs" --da_config=$ROOT_FOLDER/reproductivity/configs/raw/Amazon_Sports/aug-config.yaml

CUDA_VISIBLE_DEVICES="0, 1, 2" python $ROOT_FOLDER/parallel_run.py --model=SASRec --dataset=ml-1m --n_gpus=3 --n_task_per_gpu=2 --output_path="${OUT_FOLDER}/ml-1m-result-outs" --ckpt_path="${OUT_FOLDER}/ml-1m-ckpt-outs" --da_config=$ROOT_FOLDER/reproductivity/configs/raw/ml-1m/aug-config.yaml

CUDA_VISIBLE_DEVICES="0, 1, 2" python $ROOT_FOLDER/parallel_run.py --model=SASRec --dataset=yelp --n_gpus=3 --n_task_per_gpu=2 --output_path="${OUT_FOLDER}/yelp-result-outs" --ckpt_path="${OUT_FOLDER}/yelp-ckpt-outs" --da_config=$ROOT_FOLDER/reproductivity/configs/raw/yelp/aug-config.yaml

python $ROOT_FOLDER/da_analyzer.py --result_paths=$OUT_FOLDER
