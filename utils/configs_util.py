import json
import os.path
from os.path import join
from utils.constants import PROJECT_ROOT
import settings

settings.init()
def create_config(args, model_path):

    # use default config
    config_path = join(
            PROJECT_ROOT, settings.config_file
        )
    # Store path in config.json
    with open(config_path, 'w') as config_file:
        json.dump({'timed_dir': os.path.basename(model_path),
                   'random_seed': args.seed, 'num_train': args.num_training, 'num_iter': args.num_iter}, config_file)

def load_config():
    config_path = join(
        PROJECT_ROOT, settings.config_file
    )
    with open(config_path) as fp:
        return json.load(fp)