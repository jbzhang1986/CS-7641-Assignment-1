# Assignment-1

This package should contain all code and data necessary to run all of the expierments for assignment 1.

## Installation

1. Install Conda, a CLI tool

   https://conda.io/docs/user-guide/install/index.html

2. Use Conda to install all dependencies through `environment.yml`

```
    conda env create -f environment.yml
```

3. Activate the environment created by conda


```
    source activate cs-7641
```

4. You're set, read the next sections in order to be able to run any of the experiments.  

## Run

```
python main.py --help
height has been deprecated.

usage: main.py [-h] [-d {wine,credit_card}]
               {clean,knn,svm,ann,dt,boosting} ...

optional arguments:
  -h, --help            show this help message and exit
  -d {wine,credit_card}, --dataset {wine,credit_card}
                        Which dataset to run on

subcommands:
  {clean,knn,svm,ann,dt,boosting}
    clean               Clean the stats from original to final and show me
                        information
    knn                 Run k-nearest neighbors
    svm                 Run Support Vector Machines
    ann                 Run neural networks
    dt                  Run decision trees
    boosting            Run boosting
(cs-7641)

```


## Example

To run for example, the wine problem with an ANN, use the following command.

```
python main.py -d wine ann
```