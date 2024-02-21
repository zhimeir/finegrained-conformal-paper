# finegrained-conformal-paper
This repository contains the code to reproduce the numerical results in 
[Not all distributional shifts are equal: Fine-grained robust conformal inference](https://arxiv.org/abs/2402.13042).


## Folders
- `datasets/`: includes all the datasets for real data analysis.
    - `nslm_semi_synthetic.csv` is the semi-synthetic data generated based on the NSLM dataset of [Carvalho et al](https://arxiv.org/abs/1907.07592).
    - `covid_original.csv` and `covid_replication.csv` correspond to the results from the [original](https://journals.sagepub.com/doi/full/10.1177/0956797620939054) and 
    [replication](https://journals.sagepub.com/doi/10.1177/09567976211024535) study on the sharing COVID-related information. 
- `scripts/`: contains the python code for reproducing the simulation and real application results of the paper.
    - `simulation.py` reproduces the simulation results in Section 5.
    - `nslm.py` reproduces the results in Section 6.1.
    - `acs_income.py` reproduces the results in Sectin 6.2.
    - `covid.py` reproduces the results in Section 6.3.
- `results/`: stores all the output files.

## Getting started
To set up the virtual environment, run the following command in terminal:
```
source ./scripts/venv3115/bin/activate
```

## Running the code 
To reproduce one run of the simulation, execute the following command in terminal:
```
cd scripts
python3 simulation.py 1
```
To reproduce one run of the experiment on the NSLM dataset, execute the following command in terminal:
```
python3 nslm.py 1
```
To reproduce one run of the experiment on the ACS income dataset, execute the following command in terminal:
```
python3 acs_income.py 1
```

To reproduce one run of the experiment on the COVID dataset, execute the following command in terminal:
```
python3 covid.py 1
```

### Acknowledgment
- The code for estimating the conditional cumulative distribution function is 
from the [qosa-index](https://gitlab.com/qosa_index) package.
- The ACS income dataset 
and corresponding models are obtained from [WhyShift](https://github.com/namkoong-lab/whyshift/tree/main/whyshift).
- The COVID information datasets are from [awesome-replicability-data](https://github.com/ying531/awesome-replicability-data). 
