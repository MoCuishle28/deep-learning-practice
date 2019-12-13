import argparse

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
# dest='train' 可以通过 args.train 来访问
parser.add_argument('--train', dest='train', action='store_true', default=True)	# args.train == True
parser.add_argument('--test', dest='train', action='store_false')	# args.train == False
args = parser.parse_args()

print(args.train)