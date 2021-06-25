import argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training for iResNets')
parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
args = parser.parse_args()
print(args)