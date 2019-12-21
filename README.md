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
| versions | type | details |
| :-----| :---- | :---- |
| 0.8 | no-source | bohb: no source, only the target data |
| 0.9 | only-1-source | select one source and use it |
| 1.0 | multi-source: weight | using multi source, map #inordered-pairs int prob (softmax) |
| 1.1 | multi-source: weight | using multi source, weight each model with the probability that it is the model in the ensemble with the lowest ranking loss |
| 1.2 | multi-source: weight | using multi source, weights from SQP problem by incorporating the differentiable pairwise ranking loss function |
| 1.3 | multi-source: fusion | gpoe and independently combination |
