import os.path
import time
from utils.file_utils import FileUtils
from utils.configs_util import create_config
from sdg_finetuning import *
from arg_parser import get_args
from output import OUTPUT_DIR
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, evaluation
from warnings import simplefilter
from utils.log_utils import LogUtils
from utils.constants import PROJECT_ROOT
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def load_models(args, multi_label_model_path):

    with open(os.path.join(multi_label_model_path, f"linear_classifier_{args.dataset}.pkl"), "rb") as model_file:
        linear_classifier = pickle.load(model_file)

    with open(os.path.join(multi_label_model_path, f"mlb_{args.dataset}.pkl"), "rb") as mlb_file:
        mlb = pickle.load(mlb_file)

    embbeding_model_path = os.path.join(multi_label_model_path, f"sbert_embedding_model_{args.dataset}")
    embedding_model = SentenceTransformer(embbeding_model_path)

    return linear_classifier, mlb, embedding_model


def write_results_to_file(args, metrics_tuple, multi_label_sdg_model_path):
    (strict_accuracy, weak_accuracy, hamming, precision_micro, recall_micro, f1_micro, precision_macro,
                recall_macro, f1_macro, jaccard_sim, logloss, macro_mAP, weighted_mAP, micro_mAP, APs) = metrics_tuple
    # check if output directory exists
    OUTPUTS_DIR = os.path.join(OUTPUT_DIR, 'results')
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)

    RESULTS_DIR = os.path.join(OUTPUTS_DIR, multi_label_sdg_model_path)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    results_path = os.path.join(RESULTS_DIR, f'results_{args.dataset}_{args.seed}.txt')
    print(results_path)
    with open(results_path, 'w') as file:
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
    # Check if logging has already been configured
    if not logging.getLogger().handlers:
        LogUtils.setup_logging(log_file_path=f'{PROJECT_ROOT}/repo.log')

    logger = logging.getLogger(__name__)
    sbert_model = SentenceTransformer('all-distilroberta-v1')
    start_time = time.time()
    logger.info(f'Process started at: {start_time}')
    OUTPUTS_DIR = os.path.join(OUTPUT_DIR, 'results')
    if args.do_train:
        logger.info(f'Model training strated at: {start_time}')
        multi_label_sdg_model_path = FileUtils.create_timed_directory(MODEL_DIR, args.seed)
        create_config(args, multi_label_sdg_model_path)
        if args.label_desc_finetuning:
            logger.info('SBERT finetuning on label desc data')
            desc_model = desc_finetuning(sbert_model)
            logger.info(f'SBERT finetuning on {args.dataset} dataset')
            results, predictions = multi_label_trainer(desc_model)

        else:
            logger.info(f'SBERT finetuning on {args.dataset} dataset')
            results, predictions = multi_label_trainer(sbert_model)

        config_data = load_config()
        trained_model_dir = config_data["timed_dir"]
        PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, trained_model_dir)
        if not os.path.exists(PREDICTIONS_DIR):
            # If it doesn't exist, create the directory
            os.makedirs(PREDICTIONS_DIR)
        write_results_to_file(args, results, multi_label_sdg_model_path)
        predictions.to_csv(os.path.join(PREDICTIONS_DIR, "predictions.tsv"), sep='\t', index=False)

    if args.do_pred:
        trained_model_dir = str
        OUTPUTS_DIR = os.path.join(OUTPUT_DIR, 'results')
        config_data = load_config()
        if "timed_dir" in config_data:
            trained_model_dir = config_data["timed_dir"]
        else:
            logger.info('Check the config file. If no timed_dir, do model training first')
        multi_label_model_path = os.path.join(MODEL_DIR, os.path.basename(trained_model_dir))
        # linear_classifier = load_classifier(args, multi_label_model_path)
        # mlb = load_mlb(args, multi_label_model_path)
        # embedding_model = load_embedding_model(args, multi_label_model_path)
        linear_classifier, mlb, embedding_model = load_models(args, multi_label_model_path)
        results = sdg_prediction(linear_classifier, embedding_model, mlb)
        results_df = pd.DataFrame(results)
        PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, trained_model_dir)
        if not os.path.exists(PREDICTIONS_DIR):
            # If it doesn't exist, create the directory
            os.makedirs(PREDICTIONS_DIR)
        results_df.to_csv(os.path.join(PREDICTIONS_DIR, f'predictions_{args.dataset}_{args.seed}.txt'), sep='\t',
                          index=False, header=True)
        logger.info(f"predictions saved to {PREDICTIONS_DIR}")

    end_time = time.time()

    logger.info(f'Process ended at: {end_time}')


if __name__ == "__main__":
   main()