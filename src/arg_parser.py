import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Multi-label SDG classification")

    parser.add_argument("--num_training", default=256, type=int,
                        help="few-shot instances")
    parser.add_argument("--dataset", default="knowledge_hub", choices=["osdg", "benchmark", "knowledge_hub",
                                                                       "manually_annotated", "synthetic"],
                        help="name of the dataset used")
    parser.add_argument("--num_iter", default=5, type=int,
                        help="Number of iterations for ST fine-tuning")
    parser.add_argument("--method", default="multi_label", choices=["multi_class", "multi_label"],
                        help="value Multi-class Vs multi-label")

    parser.add_argument("--do_train", action="store_true", help="Whether to perform training")
    parser.add_argument("--do_in_domain_eval",  action="store_true", help="Whether to perform in domain evaluation")
    parser.add_argument("--do_synthetic_eval", action="store_true", help="Whether to perform synthetic test evaluation")
    parser.add_argument("--do_augmented_eval", action="store_true", help="Whether to perform in domain "
                                                                         "augmented data evaluation")
    parser.add_argument("--do_pred", action="store_true", help="Whether to perform prediction")
    #parser.add_argument("--do_eval", action="store_true", help="Whether to perform just evaluation using "
             #                                                  "different datasets")

    parser.add_argument("--label_desc_finetuning", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--multi_label_finetuning", action='store_true',
                        help="Whether to run training.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    return args
