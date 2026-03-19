#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# submit_finetune.sh
# Slurm job script for BERT fine-tuning on Purdue Anvil GPU nodes.
#
# Targets the "gpu" partition (A100 nodes, 40GB VRAM each).
# For H100 nodes (80GB), change --partition=gpu to --partition=gpu-debug
# or check available partitions with: sinfo -o "%P %G %l" | grep gpu
#
# Submit with:
#   sbatch submit_finetune.sh
#
# Monitor with:
#   squeue -u $USER
#   tail -f finetune_<jobid>.log
# ─────────────────────────────────────────────────────────────────

#SBATCH --job-name=spot-bert-finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1          # 1x A100 is plenty for this dataset size
#SBATCH --mem=32G
#SBATCH --time=00:30:00            # 30 min is generous; expect ~5-10 min on A100
#SBATCH --output=finetune_%j.log
#SBATCH --error=finetune_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jfrancisco@cpp.edu   # ← update to your actual email

# ── Environment ──────────────────────────────────────────────────
echo "============================================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "Start time  : $(date)"
echo "Working dir : $(pwd)"
echo "============================================================"

# Load Anvil's module environment
module purge
module load anaconda
module load cuda/11.8.0   # Adjust to available CUDA version on Anvil

# Activate your conda environment (create it once with setup_env.sh below)
conda activate spot-bert

# ── Verify GPU ───────────────────────────────────────────────────
echo ""
echo "[CHECK] GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ── Step 1: Generate training data ───────────────────────────────
echo "[STEP 1] Generating training data..."
python generate_training_data.py \
    --output   training_data.jsonl \
    --label-map label_map.json \
    --seed     42

echo ""

# ── Step 2: Fine-tune BERT ────────────────────────────────────────
echo "[STEP 2] Starting BERT fine-tuning..."
python finetune_bert.py \
    --data       training_data.jsonl \
    --labels     label_map.json \
    --out        ./models/bert-spot-intent \
    --base-model bert-base-uncased \
    --epochs     10 \
    --batch      32 \
    --lr         2e-5 \
    --max-len    128 \
    --val-split  0.15 \
    --seed       42

echo ""

# ── Step 3: Smoke test the saved model ───────────────────────────
echo "[STEP 3] Running smoke test on saved model..."
python finetune_bert.py \
    --smoke-test \
    --out ./models/bert-spot-intent

echo ""
echo "============================================================"
echo "Job complete: $(date)"
echo "Model saved to: ./models/bert-spot-intent/"
echo ""
echo "Transfer model to local machine with:"
echo "  scp -r x-jfrancisco@anvil.rcac.purdue.edu:\$SCRATCH/spot-finetune/models ./models"
echo "============================================================"
