#!/bin/bash 
#SBATCH --cpus-per-task=2
#SBATCH -t 0-00:10  # time (D-HH:MM)
#SBATCH --mem=32G

#SBATCH --gres=gpu:1,vram:24G
# Note: gpu_requeue has a time limit of 1 day (sinfo --Node -p gpu_requeue --Format CPUsState,Gres,GresUsed:.40,Features:.80,StateLong,Time)
#SBATCH -p gpu_quad,gpu_marks,gpu,gpu_requeue
#SBATCH --qos=gpuquad_qos
#SBATCH --mail-type=TIME_LIMIT_80,TIME_LIMIT,FAIL,ARRAY_TASKS

# Note: Directories will be created if they don't exist, relative to the path where you call the script, so make sure to call from the right location
#SBATCH --output=../slurm/msa_transformer/%x-%A_%3a_%u.out  # Nice tip: using %3a to pad to 3 characters (23 -> 023) so that filenames sort properly
#SBATCH --error=../slurm/msa_transformer/%x-%A_%3a_%u.err
#SBATCH --job-name="wt_perplexity_esm2"
#SBATCH --array=0-216  # end is inclusive. 0-based, and CSV has a header column -> take the CSV row and minus 2

set -eo pipefail # fail fully on first line failure (from Joost slurm_for_ml)
# Make prints more stable (Milad)
export PYTHONUNBUFFERED=1

echo "hostname: $(hostname)"
echo "Running from: $(pwd)"
echo "Submitted from SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"

# module load miniconda3/4.10.3
# source activate /n/groups/marks/software/anaconda_o2/envs/proteingym_env  # Source activate works with the O2 miniconda?
module load gcc/6.2.0
module load cuda/10.2

echo "Conda: $(which conda)"
echo "Python: $(which python)"
# Test python
echo "$(nvidia-smi)"
echo "Note: not testing gpu available for now"
#/n/groups/marks/software/anaconda_o2/envs/proteingym_env/bin/python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'Cuda available: {torch.cuda.is_available()}'); assert torch.cuda.is_available()"

# Important to run this from within the slurm_scripts directory where the script lives
source ../zero_shot_config.sh

# MSA transformer checkpoint 
export model_checkpoint=/n/groups/marks/projects/marks_lab_and_oatml/protein_transformer/baseline_models/MSA_transformer/esm_msa1b_t12_100M_UR50S.pt
export DMS_index=$SLURM_ARRAY_TASK_ID
# For the wt perplexity project
export dms_output_folder=/n/groups/marks/users/lood/ProteinGym/notebooks/esm_msa1b_t12_100M_UR50S_wt_pseudo_ppl/ #"${DMS_output_score_folder_subs}/MSA_Transformer/"
export scoring_strategy=wt-pseudo-ppl #masked-marginals # MSA transformer only supports "masked-marginals" #"wt-marginals"
export scoring_window="overlapping"
export model_type=MSA_transformer
# export random_seeds=(1 2 3 4 5)  # Pass these in manually, pretty easy to just let the seeds run down the clock (and then later we could launch jobs that run specific seeds if we want that granularity across more GPUs)

# ESM-2
export model_type="ESM2"
export scoring_strategy=masked-pseudo-ppl # wt-pseudo-ppl

export dms_output_folder=/n/groups/marks/users/lood/ProteinGym/notebooks/ESM2-650B/
export model_checkpoint=/n/groups/marks/projects/marks_lab_and_oatml/ProteinGym/baseline_models/ESM2/esm2_t33_650M_UR50D.pt

# Overwrite the config file here
export DMS_data_folder_subs=../../notebooks/per_wt_seq/

# Note: the directory depth has changed
/n/groups/marks/software/anaconda_o2/envs/proteingym_env/bin/python ../../proteingym/baselines/esm/compute_fitness.py \
    --model-location ${model_checkpoint} \
    --model_type ${model_type} \
    --dms_index ${DMS_index} \
    --dms_mapping ${DMS_reference_file_path_subs} \
    --dms-input ${DMS_data_folder_subs} \
    --dms-output ${dms_output_folder} \
    --scoring-strategy ${scoring_strategy} \
    --scoring-window ${scoring_window} \
    --msa-path ${DMS_MSA_data_folder} \
    --msa-weights-folder ${DMS_MSA_weights_folder} \
    --seeds 1 #2 3 4 5 #\
    # --msa-samples 10 # TMP debugging GPU/memory issues

echo "Done"
