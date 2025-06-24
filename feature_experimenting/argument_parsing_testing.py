import argparse

# Default config as a dict
DEFAULT_CONFIG = {
    "batch_size": 32,
    "lr": 0.001,
    "epochs": 10,
    "model": "resnet18",
    "data_dir": "./data",
    "save_dir": "./checkpoints"
}

def get_parser(defaults):
    parser = argparse.ArgumentParser()

    # Dynamically create arguments from defaults
    for key, value in defaults.items():
        arg_type = type(value) if value is not None else str
        parser.add_argument(f'--{key}', default=value, type=arg_type)

    return parser

def parse_config():
    parser = get_parser(DEFAULT_CONFIG)
    args = parser.parse_args()
    
    # Convert Namespace to dict and return
    config = vars(args)
    return config

if __name__ == "__main__":
    config = parse_config()
    print(config)
