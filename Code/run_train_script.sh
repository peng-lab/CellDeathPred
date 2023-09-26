#!/bin/bash

#SBATCH -o /home/haicu/alaa.bessadok/for_Alaa/logs_slurm/slurm_python_%j.job
#SBATCH -e /home/haicu/alaa.bessadok/for_Alaa/logs_slurm/slurm_python_%j.job
#SBATCH -J python

#SBATCH -p gpu_p 

##SBATCH --nodelist=supergpu02pxe
##SBATCH --nodelist=supergpu05

#SBATCH --qos gpu
#SBATCH --partition=gpu_p

#SBATCH -c 6 #CPU cores required
#SBATCH --mem=36G #Memory required

#SBATCH --gres=gpu:3

#SBATCH -t 48:00:00 #Job runtime
#SBATCH --nice=10000 #Manual priority. Do not change this.

echo $HOME
source $HOME/.bashrc
echo 'Starting python script'
source $HOME/miniconda2/bin/activate #conda 
cd /home/haicu/alaa.bessadok/for_Alaa/ # where your project is located

# activate the conda enviroment
conda activate hcs_env_3

exper_name="train_exp67_2"
#exper_name="train_exp6_29_4"
#exper_name="train_24h_pl45_107"
#exper_name="stain1_woMiner_pl13_1"

#dataset="dataset_exp5"
#dataset="dataset_exp6"
#dataset="dataset_exp6"
dataset="dataset_exp67"
#dataset="dataset_24h"
#dataset="dataset_upsampled_atp_haf_split_plates"

echo $(date '+%Y-%m-%d') >> /home/haicu/alaa.bessadok/for_Alaa/logs_slurm/list_jobs_date.txt
echo $SLURM_JOB_ID >> /home/haicu/alaa.bessadok/for_Alaa/logs_slurm/list_jobs_date.txt
echo $exper_name >> /home/haicu/alaa.bessadok/for_Alaa/logs_slurm/list_jobs_date.txt
echo $dataset >> /home/haicu/alaa.bessadok/for_Alaa/logs_slurm/list_jobs_date.txt

echo 'exp67; bn 0.1; metric 0.5 + ce 0.5; efficientnet_b0, range2; train plates 1,2,5,6,7,8,10; temp 0.1;FREEZE BN; all fields' >> /home/haicu/alaa.bessadok/for_Alaa/logs_slurm/list_jobs_date.txt
echo "" >> /home/haicu/alaa.bessadok/for_Alaa/logs_slurm/list_jobs_date.txt # empty line

# run python file 
# for stain 1, 2 dataset (old)
#python /storage/groups/peng/projects/aidin_Kenjis_project/batch_effect_removal/main.py  --seed 0 --log_path "/storage/groups/peng/projects/aidin_Kenjis_project/batch_effect_removal/logs_${exper_name}" --model_save_path "/storage/groups/peng/projects/aidin_Kenjis_project/batch_effect_removal/saved_models_${exper_name}" --tensorboard_path "/storage/groups/peng/projects/aidin_Kenjis_project/batch_effect_removal/tensorboard_${exper_name}" --stain 1 --zone 'medium' --mode 'train'  --classes 3  --ds "${dataset}" --bn_mom 0.6 --epochs 300 --embedding_size 512 --backbone 'resnet18' --miner_margin 0.549 --metric_loss_weight 1.0 --batch_size 120 --samples_per_class 20 --lr_const 1e-4 --classifier_loss_weight 0.0 --triplet_margin 0.02 --disable_emb True --crop True --avg_embs True --exper_name ${exper_name} --lr_scheduler True --train_plates 1,3 --my_miner False

# for new dataset (24h,72h)
#python /storage/groups/peng/projects/aidin_Kenjis_project/batch_effect_removal/main.py  --seed 0 --log_path "/storage/groups/peng/projects/aidin_Kenjis_project/batch_effect_removal/logs_${exper_name}" --model_save_path "/storage/groups/peng/projects/aidin_Kenjis_project/batch_effect_removal/saved_models_${exper_name}" --tensorboard_path "/storage/groups/peng/projects/aidin_Kenjis_project/batch_effect_removal/tensorboard_${exper_name}" --mode 'train' --classes 3  --ds "${dataset}" --bn_mom 0.6 --epochs 300 --embedding_size 512 --backbone 'resnet34' --miner_margin 0.549 --metric_loss_weight 1.0 --batch_size 120 --samples_per_class 20 --lr_const 1.5e-04 --classifier_loss_weight 0.0 --triplet_margin 0.02 --disable_emb True --crop True --avg_embs True --exper_name ${exper_name} --lr_scheduler True --select_plates 'a','b' --my_miner False --load_model 'stain2_myMiner_try_woMiner_pl13_5' --checkpoint 89 --stain 2

python /home/haicu/alaa.bessadok/for_Alaa/main.py  --seed 0 --log_path "/home/haicu/alaa.bessadok/for_Alaa/logs_${exper_name}" --model_save_path "/home/haicu/alaa.bessadok/for_Alaa/saved_models_${exper_name}" --tensorboard_path "/home/haicu/alaa.bessadok/for_Alaa/tensorboard_${exper_name}" --mode 'train' --classes 3  --ds "${dataset}" --bn_mom 0.1 --epochs 300 --embedding_size 1280 --backbone 'efficientnet_b0' --miner_margin 0.01 --metric_loss_weight 0.5 --batch_size 42 --samples_per_class 2 --lr_const 1.25e-5 --classifier_loss_weight 0.5 --triplet_margin 0.01 --disable_emb True --crop True --avg_embs True --exper_name ${exper_name} --lr_scheduler True --my_miner False --ds_h '24' --select_plates 1,2,5,6,7,8,10 --choose_range 'range2' --load_model 'train_exp7_1' --checkpoint 260 --stain 3

echo $SLURM_JOB_ID > /home/haicu/alaa.bessadok/for_Alaa/logs_with_${exper_name}/job_name.txt
