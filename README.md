## Hyperparameter Tuning Project
In this project, we provided a hyperparameter optimization method based Hyperband.

### Experimental Environment Installation.
Software needed: 1) anaconda, 2) python=3.5 .

Install necessary requirements:
- Softwares: `sudo apt-get install libeigen3-dev swig gfortran`.
- Python packages: `for req in $(cat requirements.txt); do pip install $req; done`.

## Docs
### Project Dirs
- mfes/facade: the implemented method and compared baselines.
- mfes/evaluate_function: objective function for each benchmark.
- test/exps: the python scripts in the experiments.
- test/plots: the scripts for drawing.

### Experiments Design
##### Exp.1: compare different versions of MFES.
Exp settings: runtime_limit=18000, n_worker=1.
Execution scripts: `python test/exps/evaluate_weight_method.py --methods 'rank_loss_p_norm-2,rank_loss_p_norm-3,rank_loss_p_norm-1,rank_loss_single,rank_loss_prob' --rep 5 --runtime_limit 18000`

| versions | type | details |
| :-----| :---- | :---- |
| 0.9 | only-1-source | select one source and use it |
| 1.0 | multi-source: weight | using multi source, map #ordered-pairs-percentage int prob (power_num=1,2,3) |
| 1.1 | multi-source: weight | using multi source, weight each model with the probability that it is the model in the ensemble with the lowest ranking loss |
| 1.2 | multi-source: fusion | gpoe and independently combination |


##### Exp.2: Result comparison on FCNet-MNIST.
Exp settings: runtime_limit=18000, n_worker=1.
Compared methods: mfes-hb, hb, bohb, lcnet-hb, smac, smac-es, random_search.


##### Exp.3: Result comparison on RESNet-CIFAR10.
Exp settings: runtime_limit=50000, n_worker=1.
Compared methods: mfes-hb, hb, bohb, lcnet-hb, smac, smac-es, random_search.


##### Exp.4: Result comparison on RESNet-CIFAR100.
Exp settings: runtime_limit=50000, n_worker=1.
Compared methods: mfes-hb, hb, bohb, lcnet-hb, smac, smac-es, random_search.


##### Exp.5: Result comparison on XGBoost-Covtype.
Exp settings: runtime_limit=27000, n_worker=1.
Compared methods: mfes-hb, hb, bohb, tse, fabolas, smac, random_search.


##### Exp.6: Parallel version evaluations on ResNet-CIFAR10.
Exp settings: runtime_limit=50000, n_worker=1.
Compared methods: mfes-hb, hb, bohb, batch-bo, random_search.
