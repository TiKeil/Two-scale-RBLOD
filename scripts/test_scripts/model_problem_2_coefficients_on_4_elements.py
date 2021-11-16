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
from gridlod.world import World, Patch
from perturbations_for_2d_data import visualize

from scripts.problems import layer_problem_1, _construct_aFine_from_mu

# parameters for the grid size
N = 2      # <-- NOTE: These number is for visualization only.
n = 256    # <-- NOTE: These number is for visualization only.

NFine = np.array([n, n])                            # n x n fine grid elements
NpFine = np.prod(NFine + 1)                         # Number of fine DoFs
NWorldCoarse = np.array([N, N])                     # N x N coarse grid elements
boundaryConditions = np.array([[0, 0], [0, 0]])     # zero Dirichlet boundary conditions
NCoarseElement = NFine // NWorldCoarse
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)  # gridlod specific class

param_min, param_max = 1, 5
aFines, aFineCoefficients, f, f_fine, model_parameters, aFine_Constructor = \
    layer_problem_1(NFine, world, coefficients=3, plot=False, return_fine=True)

"""
Plot data for the figures
"""
fullpatch = Patch(world, np.inf, 0)
aFines = aFine_Constructor(fullpatch)
for i, mu_plot in enumerate([[1,0,0], [0,1,0], [0,0,1], [1,2,3]]):
    aFine_ = _construct_aFine_from_mu(aFines, aFineCoefficients, model_parameters.parse(mu_plot))
    if i == 0 or i == 3:
        visualize.drawCoefficient_origin(NFine, aFine_, logNorm=True, colorbar_font_size=18)
    else:
        visualize.drawCoefficient_origin(NFine, aFine_, logNorm=True)
    plt.tight_layout()
    # plt.savefig(f'patch_patch_diff_oc_mp_{i}.png', bbox_inches='tight')
    plt.show()