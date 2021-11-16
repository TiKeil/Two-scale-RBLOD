```
# ~~~
# This file is part of the paper:
#
#           " An Online Efficient Two-Scale Reduced Basis Approach
#                for the Localized Orthogonal Decomposition "
#
#   https://github.com/TiKeil/Two-scale-RBLOD.git
#
# Copyright 2021 all developers. All rights reserved.
# License: licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Tim Keil
#   Stephan Rave
# ~~~
```

In this repository, we provide the code for the numerical experiments in Section 6
of the paper **"An Online Efficient Two-Scale Reduced Basis Approach for the
Localized Orthogonal Decomposition"** by Tim Keil and Stephan Rave.
The preprint is available [here](...).

For just taking a look at the experiment outputs and data, you do not need to
install the software. Just go to `scripts/test_scripts/test_outputs/`,
where we have stored printouts of our numerical experiments.
In order to relate this data to the paper, we provide further information in the next Section.

If you want to have a closer look at the implementation or generate the results by
yourself, we provide simple setup instructions for configuring your own Python environment.
We note that our setup instructions are written for Ubuntu Linux only and we do not provide
setup instructions for MacOS and Windows.
Our setup instructions have successfully been tested on a fresh Ubuntu 20.04.2.0 LTS system.
The actual experiments have been computed on the
[PALMA II HPC cluster](<https://www.uni-muenster.de/IT/en/services/unterstuetzungsleistung/hpc/index.shtml>).
For the concrete configurations we refer to the scripts in `submit_to_cluster`.

# How to quickly find the data from the paper

We provide information on how to relate the output files to the figures and tables in the paper.
All output files and figures are stored in `scripts/test_scripts/test_outputs`.
Note that the outputs are verbose outputs compared to the ones that we present in the paper,
which is also the reason why we do not provide scripts for constructing the error plots and
tables from the paper.

- Figure 1: The plots for this figure are named by `model_problem_1_coefficient_*.png`.
- Table 1: The output for all three choices of the coarse mesh are stored in 
`model_problem_1_H_8.dat`, `model_problem_1_H_16.dat`, `model_problem_1_H_32.dat`.
- Figure 2: This plot has been constructed from the printout in
`estimator_study_for_different_epsilon.txt`.
- Figure 3: All errors and estimates are stored in `estimator_study_certified.txt` for
the certified training (Figure 3A) and in `estimator_study_coarse.txt`
for the training with the coarse part of the certified estimator (Figure 3B).
- Figure 4: The plots are stored as `model_problem_2_coefficients_*.png`.
- Table 2: This table has been filled with the data from `model_problem_2.dat`.

# Organization of the repository

We used several external software packages:

- [pyMOR](https://pymor.org) is a software library for Model Order Reduction.
- [gridlod](https://github.com/fredrikhellman/gridlod) is a discretization toolkit for the
Localized Orthogonal Decompostion (LOD) method. 
- [perturbations-for-2d-data](https://github.com/TiKeil/perturbations-for-2d-data) contains
a coefficient generator for constructing randomized and highly oscillating coefficients.

We added the external software as editable submodules with a fixed commit hash.
For the TSRBLOD, we have developed a Python module `rblod`.
The rest of the code is contained in `scripts`, where you find the definition of the model problems
(in `problems.py`) and the main scripts for the numerical experiments.
The file `run_experiment.py` contains a lot of additional documentation for understanding
the specific implementation of our methods.

# Setup

On a standard Ubuntu system (with Python and C compilers installed) it will most likely be enough
to just run our setup script. For that, please clone the repository

```
git clone https://github.com/TiKeil/Two-scale-RBLOD.git TSRBLOD
```

and execute the provided setup script via 

```
cd TSRBLOD
./setup.sh
```

If this does not work for you, and you don't know how to help yourself,
please follow the extended setup instructions below.

## Installation on a fresh system

We also provide setup instructions for a fresh Ubuntu system (20.04.2.0 LTS).
The following steps need to be taken:

```
sudo apt update
sudo apt upgrade
sudo apt install git
sudo apt install build-essential
sudo apt install python3-dev
sudo apt install python3-venv
sudo apt install libopenmpi-dev
sudo apt install libsuitesparse-dev
```

Now you are ready to clone the repository and run the setup script:

```
git clone https://github.com/TiKeil/Two-scale-RBLOD.git TSRBLOD
cd TSRBLOD
./setup.sh
```

# Running the experiments

You can make sure your that setup is complete by running the minimal test script
```
cd scripts/test_scripts
./minimal_test.sh
```

If this works fine (with timings and max error output in the end), your setup is working well.

If you are interested in the exact shell scripts that we have used for the numerical experiments
in the paper, please have a look at `scripts/test_scripts`. Moreover, we provide further information
on how to reconstruct the figures and tables from the paper.
Please note that these shell scripts will produce verbose outputs.
The above mentioned output files in `scripts/test_scripts/test_outputs` are a minimal version of this.
For executing Python scripts, you need to activate the virtual environment by

```
source venv/bin/activate
```

- Figure 1: For visualizing the diffusion coefficients, you can run the file
`model_problem_1_coefficients.py` with Python.
- Table 1: `model_problem_1.sh` executes three experiments with changing coarse mesh.
- Figure 2: This plot has been constructed from the verbose output of
`estimator_study_for_different_epsilon.sh`.
- Figure 3: The corresponding data can be obtained by running two scripts.
With `estimator_study_prepare.sh`, you can build the Stage 1 model once
(which will then be stored on disc). After the data of Stage 1 has been stored, with
`estimator_study_*.sh`, you can then compute the errors for different basis sizes.
- Figure 4: The coefficients can be obtained by calling `model_problem_2_coefficients_on_4_elements.py`
with Python. Note that this is not the actual diffusion coefficient but rather a way to visualize it on
4 elements.
- Table 2: This data has been generated by `model_problem_2.sh`.

Note that especially for `model_problem_2.sh`, an HPC cluster is required.
In particular, starting the shell scripts with only a few parallel cores (or even without `mpirun`)
on your local computer may take days to weeks.

Please have a look at the description of the command line arguments of `scripts/run_experiment.py`
to try different configurations of the given problem classes. Note that it is also possible to solve
your own parameterized problems with our code since the problem definitions that are used in
`scripts/problems.py` are very general. 

# Additional information on the diffusion coefficients in Section 6.2

In the paper, we have not provided an expression for the diffusion coefficient that has been used for
test case 2. This is due to the fact that the coefficient is a random diffusion tensor with random noise
(also for the periodic parts). The seed for the random distribution is defined by the index of the
coarse element T. The concrete coefficients are constructed by the coefficient generator `buildcoef2d`
from [perturbations-for-2d-data](https://github.com/TiKeil/perturbations-for-2d-data).
For this we refer to the function `construct_coefficients_on_T` which is defined in `scripts/problems.py`.
The class arguments uniquely determine these coefficients and also fix the random seed. With this,
the coefficients can always be reconstructed from the returned `numpy.array`,
where each entry corresponds to the constant value of the coefficient on each fine grid element.

# Questions

If there are any questions of any kind, please contact us via <tim.keil@wwu.de>.
