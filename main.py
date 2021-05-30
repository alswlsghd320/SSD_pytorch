import configparser
import argparse
from agent.train import train
from agent.eval import * #eval

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--config', type=str, default='config.ini')

if __name__ == '__main__':
    args = parser.parse_args()
    config_path = args.config_path

    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')

    if args.train is True:
        cfg = config['Train']
        train(cfg)
    elif args.eval is True:
        cfg = config['Test']
        eval(cfg)