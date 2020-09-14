## Hyperparameter Tuning Project
In this project, we provided a hyperparameter optimization method based on Hyperband.

### Experimental Environment Installation.
Software needed: 1) anaconda, 2) python>=3.5.

Install necessary requirements:
- Softwares: `sudo apt-get install libeigen3-dev swig3.0 gfortran`.
- Run `ln -s /usr/bin/swig3.0 /usr/bin/swig`
- Python packages: `for req in $(cat requirements.txt); do pip install $req; done`.

## Docs
### Project Dirs
- mfes/facade: the implemented method and compared baselines.
- mfes/evaluate_function: objective function for each benchmark.
- test/exps: the python scripts in the experiments.
- test/plots: the scripts for drawing.

### Experiments Design
See test/exps/evaluate_compared_baseline.py to get the name of each baseline method.

##### Exp.1: compare different versions of MFES.
Exp settings: runtime_limit=18000, n_worker=1, rep=5.
Execution scripts: `python test/exps/evaluate_weight_method.py --methods 'rank_loss_p_norm-2,rank_loss_p_norm-3,rank_loss_p_norm-1,rank_loss_single,rank_loss_prob' --rep 5 --runtime_limit 18000`

| versions | type | details |
| :-----| :---- | :---- |
| 0.9 | only-1-source | select one source and use it |
| 1.0 | multi-source: weight | using multi source, map #ordered-pairs-percentage int prob (power_num=1,2,3) |
| 1.1 | multi-source: weight | using multi source, weight each model with the probability that it is the model in the ensemble with the lowest ranking loss |
| 1.2 | multi-source: fusion | gpoe and independently combination |


##### Exp.2: Result comparison on FCNet-MNIST.
Exp settings: runtime_limit=18000, n_worker=1, rep=10.

Compared methods: mfes-hb, hb, bohb, lcnet-hb, smac, smac-es, random_search.

Script: `python test/exps/evaluate_compared_baseline.py --runtime_limit 18000 --benchmark fcnet --baseline your_choice --rep_num 10`


##### Exp.3: Result comparison on RESNet-CIFAR10.
Exp settings: runtime_limit=50000, n_worker=1, rep=10, R=81.

Compared methods: mfes-hb, hb, bohb, lcnet-hb, smac, smac-es, random_search.

Script: `python test/exps/evaluate_compared_baseline.py --runtime_limit 50000 --benchmark cifar --baseline your_choice --rep_num 10`

##### Exp.4: Result comparison on XGBoost-Covtype.
Exp settings: runtime_limit=27000, n_worker=1, rep=10, R=27.

Compared methods: mfes-hb, hb, bohb, tse, fabolas, smac, random_search.

Data path: data/covtype/covtype.data

Script: `python test/exps/evaluate_compared_baseline.py --runtime_limit 27000 --benchmark covtype --R 27 --baseline your_choice --rep_num 10`


##### Exp.5: Parallel version evaluations on ResNet-CIFAR10. Set the fraction in tf_gpu_option to be less than 0.33
Exp settings: runtime_limit=50000, n_worker=3, rep=10.

Compared methods: mfes-hb, hb, bohb, batch-bo, random_search.

Script: `python test/exps/evaluate_compared_baseline.py --runtime_limit 50000 --benchmark cifar --baseline your_choice --rep_num 10 --n 3`

##### Exp.6: Result comparison on System.
Exp setting: runtime_limit=14400, n_worker=1, rep=10, R=27

Compared methods: mfes-hb, smac

Data path: data/cls_datasets/example.csv (The required csv name is written in each evaluate function.)

Script: `python test/exps/evaluate_compared_baseline.py --runtime_limit 14400 --benchmark sys_letter --R 27 --baseline your_choice --rep_num 10`
