import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Multi-label SDG classification")

    parser.add_argument("--num_training", default=128, type=int,
                        help="few-shot instances")
    parser.add_argument("--dataset", default="knowledge_hub", choices=["osdg", "benchmark", 'knowledge_hub'],
                        help="name of the dataset used")
    parser.add_argument("--num_iter", default=5, type=int,
                        help="Number of iterations for ST fine-tuning")
    parser.add_argument("--method", default="multi_label", choices=["multi_class", "multi_label"],
                        help="value Multi-class Vs multi-label")
    parser.add_argument("--label_desc_finetuning", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--multi_label_finetuning", action='store_true',
                        help="Whether to run training.")
    parser.add_argument('--seed', type=int, default=100, help="random seed for initialization")

    args = parser.parse_args()

    return args
