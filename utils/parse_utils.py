import json
from argparse import ArgumentParser

def load_parse(f_name):
    if not '.' in f_name:
        f_name += '.json'
    parser = ArgumentParser()
    args = parser.parse_args()
    with open(f_name, 'r') as f:
        args.__dict__ = json.load(f)
    return args

def save_parse(f_name, args):
    if not '.' in f_name:
        f_name += '.json'
    with open(f_name, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ =='__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--resume', type=str, default='a/b/c.ckpt')
    parser.add_argument('--surgery', type=str, default='190', choices=['190', '417'])
    args = parser.parse_args()

    # save_parse("test", args)
    # args_t = load_parse("test")
    print(args)
    print(args_t)