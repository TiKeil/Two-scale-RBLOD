# ~~~
# This file is part of the paper:
#
#           " An Online Efficient Two-Scale Reduced Basis Approach
#                for the Localized Orthogonal Decomposition "
#
#   https://github.com/TiKeil/Two-scale-RBLOD.git
#
# Copyright 2019-2021 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Tim Keil (2019 - 2021)
# ~~~

import matplotlib.pyplot as plt
import numpy as np
from gridlod.world import World
from perturbations_for_2d_data import visualize

from scripts.problems import model_problem_1, _construct_aFine_from_mu

# parameters for the grid size
N = 2
n = 256

NFine = np.array([n, n])                            # n x n fine grid elements
NpFine = np.prod(NFine + 1)                         # Number of fine DoFs
NWorldCoarse = np.array([N, N])                     # N x N coarse grid elements
boundaryConditions = np.array([[0, 0], [0, 0]])     # zero Dirichlet boundary conditions
NCoarseElement = NFine // NWorldCoarse
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)  # gridlod specific class

param_min, param_max = 0, 5
aFines, aFineCoefficients, f, f_fine, model_parameters, aFine_Constructor = \
    model_problem_1(NFine, world, plot=False, return_fine=True)

# standard parameter space
verification_set = [1.8727, 2.904, 4.7536]

"""
Plot data for the figures
"""
for mu_ver in verification_set:
    mu_ver = model_parameters.parse(mu_ver)
    plt.figure("coefficient for mu")
    aFine = _construct_aFine_from_mu(aFines, aFineCoefficients, mu_ver)
    visualize.drawCoefficient_origin(NFine, aFine, logNorm=True, colorbar_font_size=16,
                                     lim=[0.8,14])
    # plt.savefig(f'full_diff_mp_1_{mu_ver.to_numpy()[0]*10000:.0f}.png', bbox_inches='tight')
    plt.show()