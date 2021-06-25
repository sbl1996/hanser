import argparse

parser = argparse.ArgumentParser(description='Train a model for image classification.')

parser.add_argument('--deep-stem', action='store_true',
                    help='use deep-stem.')
opt = parser.parse_args()
print(opt.deep_stem)