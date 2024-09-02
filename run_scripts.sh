#!/bin/bash

# Get the directory path of the current script
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


# Set PYTHONPATH environment variable
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# script to read oro.tsv and extract metadata (title, abstract) for all the ids.
python3 "$PROJECT_DIR/src/metadata_extraction.py"

# script to train/predict SDGs
python3 "$PROJECT_DIR/src/multi_label_sdg.py" --label_desc_finetuning --multi_label_finetuning --do_pred