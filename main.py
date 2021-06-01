import argparse
from agent.train import train
#from agent.eval import eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')

    args = parser.parse_args()

    if args.train is True:
        train()
    # elif args.eval is True:
    #     eval()