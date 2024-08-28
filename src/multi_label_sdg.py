import os.path
from sdg_finetuning import *
from arg_parser import get_args
from output import OUTPUT_DIR
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, evaluation
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import numpy as np

import torch
import random
import torch

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

def write_results_to_file(args, metrics_tuple):
    (strict_accuracy, weak_accuracy, hamming, precision_micro, recall_micro, f1_micro, precision_macro,
                recall_macro, f1_macro, jaccard_sim, logloss, macro_mAP, weighted_mAP, micro_mAP, APs) = metrics_tuple
    # check if output directory exists
    results_dir = os.path.join(OUTPUT_DIR, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file_path = os.path.join(results_dir, f'results_{args.seed}.txt')

    with open(file_path, 'w') as file:
        file.write("Test data Results:" + '\n')
        file.write(f"Strict Accuracy: {strict_accuracy}\n")
        file.write(f"Weak Accuracy: {weak_accuracy}\n")
        file.write(f"Hamming: {hamming}\n")
        file.write(f"Precision (Micro): {precision_micro}\n")
        file.write(f"Recall (Micro): {recall_micro}\n")
        file.write(f"F1 Score (Micro): {f1_micro}\n")
        file.write(f"Precision (Macro): {precision_macro}\n")
        file.write(f"Recall (Macro): {recall_macro}\n")
        file.write(f"F1 Score (Macro): {f1_macro}\n")
        file.write(f"Jaccard: {jaccard_sim}\n")
        file.write(f"Log loss: {logloss}\n")

        file.write(f"macro Mean Average Precision: {macro_mAP}\n")
        file.write(f"weighted Mean Average Precision: {weighted_mAP}\n")
        file.write(f"micro Average Precision: {micro_mAP}\n")
        file.write(f"Average Precision: {APs}\n")

def main():
    args = get_args()
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    if args.method == 'multi_label':
        if args.label_desc_finetuning:
            desc_model = desc_finetuning(sbert_model)
            results = multi_label_classification(desc_model)

        else:
            results = multi_label_classification(sbert_model)

        write_results_to_file(args, results)

if __name__ == "__main__":
   main()