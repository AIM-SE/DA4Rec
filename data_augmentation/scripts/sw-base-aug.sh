
SHELL_FOLDER=$(cd "$(dirname "$0")"; pwd)
ROOT_FOLDER=$SHELL_FOLDER/..
OUT_FOLDER=$ROOT_FOLDER/outs

echo "This gonna take lots of time, which is subject to your devices"

CUDA_VISIBLE_DEVICES="0, 1, 2" python $ROOT_FOLDER/parallel_run.py --model=SASRec --dataset=Amazon_Beauty --n_gpus=3 --n_task_per_gpu=2 --aug_base=recbole-slide-window --output_path="${OUT_FOLDER}/Beauty-sw50-result-outs" --ckpt_path="${OUT_FOLDER}/Beauty-sw50-ckpt-outs" --da_config=$ROOT_FOLDER/reproductivity/configs/aug_base/Amazon_Beauty/aug-config.yaml

CUDA_VISIBLE_DEVICES="0, 1, 2" python $ROOT_FOLDER/parallel_run.py --model=SASRec --dataset=Amazon_Sports --n_gpus=3 --n_task_per_gpu=2 --aug_base=recbole-slide-window --output_path="${OUT_FOLDER}/Sports-sw50-result-outs" --ckpt_path="${OUT_FOLDER}/Sports-sw50-ckpt-outs" --da_config=$ROOT_FOLDER/reproductivity/configs/aug_base/Amazon_Sports/aug-config.yaml

CUDA_VISIBLE_DEVICES="0, 1, 2" python $ROOT_FOLDER/parallel_run.py --model=SASRec --dataset=ml-1m --n_gpus=3 --n_task_per_gpu=2 --aug_base=recbole-slide-window --output_path="${OUT_FOLDER}/ml-1m-sw200-result-outs" --ckpt_path="${OUT_FOLDER}/ml-sw50-1m-ckpt-outs" --da_config=$ROOT_FOLDER/reproductivity/configs/aug_base/ml-1m/aug-config.yaml

CUDA_VISIBLE_DEVICES="0, 1, 2" python $ROOT_FOLDER/parallel_run.py --model=SASRec --dataset=yelp --n_gpus=3 --n_task_per_gpu=2 --aug_base=recbole-slide-window --output_path="${OUT_FOLDER}/yelp-sw50-result-outs" --ckpt_path="${OUT_FOLDER}/yelp-sw50-ckpt-outs" --da_config=$ROOT_FOLDER/reproductivity/configs/aug_base/yelp/aug-config.yaml

python $ROOT_FOLDER/da_analyzer.py --result_paths=$OUT_FOLDER
