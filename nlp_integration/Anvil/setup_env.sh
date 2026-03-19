#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# setup_env.sh
# Run ONCE on Anvil login node to create the conda environment.
#
# Usage:
#   bash setup_env.sh
# ─────────────────────────────────────────────────────────────────

module purge
module load anaconda

echo "[INFO] Creating conda environment: spot-bert"
conda create -y -n spot-bert python=3.10

conda activate spot-bert

echo "[INFO] Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "[INFO] Installing HuggingFace + training dependencies..."
pip install \
    transformers==4.40.0 \
    accelerate \
    datasets \
    scikit-learn \
    numpy \
    tqdm

echo ""
echo "[OK] Environment 'spot-bert' is ready."
echo "     Pre-download BERT weights now (do this on login node, not in job):"
echo ""
echo "     python -c \"from transformers import BertTokenizerFast, BertForSequenceClassification; \\"
echo "       BertTokenizerFast.from_pretrained('bert-base-uncased'); \\"
echo "       BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)\""
echo ""
echo "     This caches the model in ~/.cache/huggingface so the Slurm job"
echo "     doesn't need internet access during training."
