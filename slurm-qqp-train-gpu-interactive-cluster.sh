#!/bin/bash
#SBATCH --job-name=train_qqp
#SBATCH --output=train_qqp.txt
#SBATCH	--ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=20g
#SBATCH --gres=gpu:4
#SBATCH --qos=normal
#SBATCH --partition=rtx6000

# mitigates activation problems
eval "$(conda shell.bash hook)"

# activate the correct environment
conda activate /h/fraboeni/anaconda3/envs/py35

echo "Job started at $(date)"

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=4

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path bert-base-uncased \
    --train_file /ssd003/home/fraboeni/data/qqp/qqp_train.csv \
    --output_dir /ssd003/home/fraboeni/models/nlp-stealing/my-sup-simcse-bert-base-uncased-qqp \
    --num_train_epochs 5 \
    --per_device_train_batch_size 128 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
echo "Job ended at $(date)"