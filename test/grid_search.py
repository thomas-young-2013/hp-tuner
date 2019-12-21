import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['local', 'server'], default='local')
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--n', type=int, default=12)
parser.add_argument('--case', type=int, default=4)

args = parser.parse_args()

if args.mode == 'local':
    sys.path.append('/home/thomas/Desktop/codes/hp-tuner')
else:
    sys.path.append('/home/daim/thomas/running3/hp-tuner')

maximal_iter = args.R
n_work = args.n
case = args.case

from mfes.facade.gs import GridSearch
from mfes.evaluate_function.eval_grid_search_tf import train
from mfes.evaluate_function.eval_grid_search1_tf import train1

if __name__ == "__main__":
    # gs = GridSearch(train, maximal_iter, n_workers=n_work, case=2)
    # gs.run()
    gs = GridSearch(train1, maximal_iter, n_workers=n_work, case=5)
    gs.run()
