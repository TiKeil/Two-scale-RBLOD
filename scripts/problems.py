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
#   Stephan Rave 
#   Tim Keil     
# ~~~

import numpy as np

from gridlod import util
from gridlod.world import Patch

from perturbations_for_2d_data import buildcoef2d

import matplotlib.pyplot as plt

from pymor.parameters.functionals import (
    ExpressionParameterFunctional,
    ProjectionParameterFunctional,
)
from pymor.parameters.base import Parameters
from pymor.operators.constructions import ZeroOperator, LincombOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from perturbations_for_2d_data import visualize


def _construct_aFine_from_mu(aFines, aFinesCoefficients, mu):
    coefs = [c(mu) for c in aFinesCoefficients]
    dim_array = aFines[0].ndim
    if dim_array == 3:
        a = np.einsum('ijkl,i', aFines, coefs)
    elif dim_array == 1:
        a = np.einsum('ij,i', aFines, coefs)
    return a

def construct_coefficients_on_T(Tpatch):
    T = Tpatch.TInd
    NFine = Tpatch.NPatchFine
    bg = 0.1
    factor = 1
    bg_values = [0.08, 0.12]
    val_values = [1, 1.2]

    buildClass = buildcoef2d.Coefficient2d(
        NFine,
        bg=bg,
        val=1,
        thick=1 * factor,
        space=1 * factor,
        probfactor=5,
        right=1,
        down=1,
        diagr1=1,
        diagr2=1,
        diagl1=1,
        diagl2=1,
        LenSwitch=[4 * factor, 5 * factor, 6 * factor],
        thickSwitch=[3 * factor, 4 * factor],
        BoundarySpace=False,
        normally_distributed_bg=bg_values,
        normally_distributed_val=val_values,
        seed=T
    )
    a_1 = buildClass.BuildCoefficient()
    buildClass = buildcoef2d.Coefficient2d(
        NFine,
        bg=bg,
        val=1,
        length=4 * factor,
        thick=4 * factor,
        probfactor=1,
        space=2 * factor,
        right=1,
        BoundarySpace=True,
        normally_distributed_bg=bg_values,
        normally_distributed_val=val_values,
        seed=T
    )
    a_2 = buildClass.BuildCoefficient()
    CoefClass = buildcoef2d.Coefficient2d(
        NFine,
        bg=bg,
        val=1,
        length=2 * factor,
        thick=2 * factor,
        space=2 * factor,
        probfactor=1,
        equidistant=True,
        ChannelHorizontal=True,
        BoundarySpace=False,
        normally_distributed_bg=bg_values,
        normally_distributed_val=val_values,
        seed=T
    )
    a_3 = CoefClass.BuildCoefficient()
    return [a_1, a_2, a_3]


def construct_coefficients_model_problem_2(patch):
    coarse_indices = patch.coarseIndices
    coarse_indices_mod = coarse_indices % patch.world.NWorldCoarse[0]
    mod_old = -1
    j, l = 0, -1
    blocklists = [[], [], []]
    for i, (T, Tmod) in enumerate(zip(coarse_indices, coarse_indices_mod)):
        if Tmod < mod_old:
            j += 1
            l = 0
        else:
            l += 1
        Tpatch = Patch(patch.world, 0, T)
        # fipd = patch.world.NCoarseElement[0]
        a = construct_coefficients_on_T(Tpatch)
        for k, a_q in enumerate(a):
            # a_q = a_.reshape((fipd,fipd))
            if l==0:
                blocklists[k].append(([a_q]))
            else:
                blocklists[k][j].append(a_q)
        mod_old = Tmod
    aPatchblock = [np.block(blocklist).ravel() for blocklist in blocklists]
    return aPatchblock


def construct_coefficients_from_coords_model_problem_1(patch):
    xt = util.computePatchtCoords(patch)
    NtFine = xt.shape[0]

    Aeye = np.tile(np.eye(2), [NtFine, 1, 1])

    eps = 0.1
    a_1 = Aeye.copy()
    a_2 = Aeye.copy()
    a_3 = Aeye.copy()
    a_4 = Aeye.copy()

    # a_1
    a_1_coef_0 = lambda x: 5 * 1 / (np.pi ** 2) * 1 / (4 + 2 * np.cos(2 * np.pi * x[..., 0] / eps))
    a_1_coef0 = np.array(a_1_coef_0(xt)).T
    e_1 = np.tile(np.array([1, 0]), [NtFine, 1])
    a_1_coef0 = np.einsum("ti, t -> ti", e_1, a_1_coef0)
    a_1_ll = np.einsum("tji, ti -> tji", a_1, a_1_coef0)

    a_1_coef_1 = lambda x: 1 / (4 * np.pi) * (5 + 2.5 * np.cos(2 * np.pi * x[..., 0] / eps))
    a_1_coef1 = np.array(a_1_coef_1(xt)).T
    e_2 = np.tile(np.array([0, 1]), [NtFine, 1])
    a_1_coef1 = np.einsum("ti, t -> ti", e_2, a_1_coef1)
    a_1_ur = np.einsum("tji, ti -> tji", a_1, a_1_coef1)

    a_1 = a_1_ur + a_1_ll

    # a_2
    const = np.ones(NtFine, dtype=np.float64) * 1 / 100.0
    a_2_coef_ = lambda x: 10 + 9 * np.sin(2 * np.pi * np.sqrt(2 * x[..., 0]) / eps) * np.sin(
        4.5 * np.pi * x[..., 1] ** 2 / eps
    )
    a_2_coef = np.array(a_2_coef_(xt)).T
    a_2_ = np.einsum("tji, t -> tji", a_2, const)
    a_2 = np.einsum("tji, t -> tji", a_2_, a_2_coef)

    # a_3
    const = np.ones(NtFine, dtype=np.float64) * 1.0
    g_eps = lambda x: np.sin(
        np.floor(x[..., 0] + x[..., 1]) + np.floor(x[..., 0] / eps) + np.floor(x[..., 1] / eps)
    ) + np.cos(np.floor(x[..., 1] - x[..., 0]) + np.floor(x[..., 0] / eps) + np.floor(x[..., 1] / eps))
    a_3_coef_ = lambda x: (3 / 25.0) + 1 / 20.0 * g_eps(x)
    a_3_coef = np.array(a_3_coef_(xt)).T
    a_3_ = np.einsum("tji, t -> tji", a_3, const)
    a_3 = np.einsum("tji, t -> tji", a_3_, a_3_coef)

    # a_4
    def c_func(x):
        sum = 0
        for j in range(5):
            for i in range(j + 1):
                sum += (2 / (j + 1)) * np.cos(
                    np.floor(i * x[..., 1] - x[..., 0] / (1 + i))
                    + np.floor(i * x[..., 0] / eps)
                    + np.floor(x[..., 1] / eps)
                )
        sum *= 1 / 10.0
        sum += 1.0
        return sum

    c_eps = c_func(xt)

    def h_func(t):
        if 0.5 < t < 1:
            return t ** 4
        elif 1 < t < 1.5:
            return t ** (3 / 2)
        else:
            return t

    h_func_vectorized = np.vectorize(h_func)
    a_4_const = h_func_vectorized(c_eps)
    a_4 = np.einsum("tji, t -> tji", a_4, a_4_const)

    return a_1, a_2, a_3, a_4

def model_problem_1(NFine, world, plot=False, return_fine=True):
    """
    Definition of the problem. Build an affine decomposition from model problem 1
    """
    if return_fine:
        patch = Patch(world, np.inf, 0)
        a_1, a_2, a_3, a_4 = construct_coefficients_from_coords_model_problem_1(patch)

    # coefficients
    aFineCoefficients = [
        ExpressionParameterFunctional("2 + sin(4*mu)", {"mu": 1}),
        ExpressionParameterFunctional("2 + mu**2 - cos(sqrt(abs(mu)))", {"mu": 1}),
        ExpressionParameterFunctional("2 + cos(sqrt(abs(mu)))", {"mu": 1}),
        ExpressionParameterFunctional("1 + sqrt(abs(mu)) + (1/10.)*abs(mu)**(2/3.)", {"mu": 1}),
    ]

    model_parameters = Parameters(mu=1)

    if return_fine:
        aFines = [a_1, a_2, a_3, a_4]
        # fine right hand side
        f_fine = np.ones(world.NpFine)
    else:
        aFines = None
        f_fine = None
    # coarse right hand side
    f = np.ones(world.NpCoarse)

    if plot and return_fine:
        plt.figure("1")
        visualize.drawCoefficient_origin(NFine, a_1)
        # plt.savefig('mp_1_1.png', bbox_inches='tight')
        plt.figure("2")
        visualize.drawCoefficient_origin(NFine, a_2)
        # plt.savefig('mp_1_2.png', bbox_inches='tight')
        plt.figure("3")
        visualize.drawCoefficient_origin(NFine, a_3)
        # plt.savefig('mp_1_3.png', bbox_inches='tight')
        plt.figure("4")
        visualize.drawCoefficient_origin(NFine, a_4)
        # plt.savefig('mp_1_4.png', bbox_inches='tight')

    return aFines, aFineCoefficients, f, f_fine, model_parameters,\
           construct_coefficients_from_coords_model_problem_1


def layer_problem_1(NFine, world, coefficients=3, plot=False, return_fine=True):
    """
    Definition of the problem. Build an affine decomposition from model problem 1
    """

    model_parameters = Parameters(mu_1=1, mu_2=1, mu_3=1)

    aFineCoefficients = [
        ProjectionParameterFunctional("mu_1", 1, 0),
        ProjectionParameterFunctional("mu_2", 1, 0),
        ProjectionParameterFunctional("mu_3", 1, 0),
    ]

    if return_fine:
        fullpatch = Patch(world, np.inf, 0)
        aFines = construct_coefficients_model_problem_2(fullpatch)
        f_fine = np.ones(world.NpFine)
    else:
        f_fine = None
        aFines = None

    # right hand side
    f = np.ones(world.NpCoarse)

    return aFines, aFineCoefficients, f, f_fine, model_parameters, construct_coefficients_model_problem_2


def dummy_problem(NFine, world, mu):
    """
    Definition of a dummy problem.
    """
    a_const = np.ones(world.NtFine, dtype=np.float64) * 1.0

    # coefficients
    aFineCoefficients = [ExpressionParameterFunctional("0*mu + 1", {"mu": 1})]

    zero = ZeroOperator(NumpyVectorSpace(1), NumpyVectorSpace(1))
    ops = [zero for i in range(1)]
    lincomb = LincombOperator(ops, aFineCoefficients)

    mu = lincomb.parameters.parse(mu)
    eval = lincomb.evaluate_coefficients(mu)

    aFine = eval[0] * a_const
    aFines = [a_const]

    # right hand side
    f = np.ones(world.NpCoarse)
    f_fine = np.ones(world.NpFine)
    return aFine, aFines, aFineCoefficients, f, f_fine, lincomb.parameters
