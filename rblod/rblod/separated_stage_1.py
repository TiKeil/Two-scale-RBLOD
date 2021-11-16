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
from time import perf_counter

from gridlod import util, lod, interp, fem

from pymor.models.basic import StationaryModel
from pymor.operators.interface import Operator
from pymor.operators.constructions import VectorOperator, LincombOperator, ConstantOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from rblod.parameterized_stage_1 import ProductOperator

from pymor.reductors.coercive import CoerciveRBReductor
from pymor.algorithms.greedy import rb_greedy
from rblod.optimized_rom import OptimizedNumpyModelStage1

"""
STAGE 1
"""

def build_separated_patch_models(patch, aFineCoefficients, coercivity_estimator, training_set, atol_patch,
                                 save_correctors, aFine_Constructor, store=False, return_minimal=False, path=''):
    """
    prepare separated patch models for classic Henning RBLOD approach
    """
    tic = perf_counter()
    print(".", end="", flush=True)
    rom_sizes, optimized_roms, times, max_errs, max_err_mus, bases = [], [], [], [], [], []
    extension_failed = False
    aPatch = aFine_Constructor(patch)
    for dof in range(4):
        m = SeparatedCorrectorProblem(patch, dof, aPatch, aFineCoefficients)
        reductor = CoerciveRBReductor(m, product=m.products["h1"], coercivity_estimator=coercivity_estimator)
        greedy_data = rb_greedy(
            m, reductor, training_set=training_set, atol=atol_patch, extension_params={"method": "gram_schmidt"}
        )
        optimized_rom = OptimizedNumpyModelStage1(greedy_data["rom"], m.Kij)
        optimized_roms.append(optimized_rom.minimal_object())
        rom_sizes.append(greedy_data["rom"].operator.source.dim)
        bases.append(reductor.bases["RB"]) if save_correctors else bases.append(None)
        max_errs.append(greedy_data["max_errs"])
        max_err_mus.append(greedy_data["max_err_mus"])
        extension_failed_here = False if greedy_data["max_errs"][-1] < atol_patch else True
        if extension_failed_here:
            extension_failed = True
    del optimized_rom, reductor, greedy_data, m, aPatch
    print("s", end="", flush=True)
    walltime = perf_counter()-tic
    if store:
        np.savez(f'{path}mpi_storage/he_{patch.TInd}', rom=optimized_roms, time=walltime)
        return True
    elif return_minimal:
        return np.array([optimized_roms, walltime])
    else:
        return np.array([optimized_roms, walltime, rom_sizes, max_errs, max_err_mus, extension_failed, bases])

def henning_RBLOD_approach(optimized_rom_, mu):
    """
    single online solve for Henning RBLOD version of stage 1
    """
    Kmsij, u_roms = [], []
    for dof in range(4):
        Kmsij_from_rom_opt, u_rom = optimized_rom_[dof].output(mu, return_solution=True)
        Kmsij.append(Kmsij_from_rom_opt[0])
        u_roms.append(u_rom.to_numpy())
    Kij_constant = optimized_rom_[0].Kij_constant(mu)
    full_Kmsij = np.column_stack(Kmsij).flatten() + Kij_constant
    return full_Kmsij, u_roms


class SeparatedCorrectorProblem(StationaryModel):
    def __init__(self, patch, dof, aFinePatches, aFineCoefficients):
        self.patch = patch
        self.world = patch.world
        self.dof = dof

        As, rhsss = zip(*(self._assemble_for_aFine(aFine) for aFine in aFinePatches))
        As_ = [NumpyMatrixOperator(A) for A in As]
        rhsss = [[As_[0].range.make_array(r) for r in rhss] for rhss in rhsss]
        operator = LincombOperator(As_, aFineCoefficients)

        rhs_operators = []
        for i in range(len(aFineCoefficients)):
            rhs = rhsss[i][dof]
            rhs_op = VectorOperator(rhs)
            rhs_operators.append(rhs_op)
        rhs = LincombOperator(rhs_operators, aFineCoefficients)

        Kij_operators = []
        Kij_range = NumpyVectorSpace(4 * patch.NpCoarse)
        for aPatch in aFinePatches:
            csi = lod.computeSeparatedBasisCoarseQuantities(
                self.patch,
                [operator.source.zeros().to_numpy().ravel() for i in range(4)],
                aPatch,
            )
            Kij_operators.append(ConstantOperator(Kij_range.make_array(csi.Kmsij.ravel()), NumpyVectorSpace(1)))
        self.Kij = LincombOperator(Kij_operators, aFineCoefficients)  # for the two scale matrix

        ### prepare output operator
        outer_operators = []
        outer_coefficients = []
        for aPatch in aFinePatches:
            outer_operators.append(CorrectorProblemOutput_for_a_DoF(patch, aPatch, operator.source))
        outer_coefficients.extend(aFineCoefficients)
        output = LincombOperator(outer_operators, outer_coefficients)

        h1_matrix = fem.assemblePatchMatrix(patch.NPatchFine, patch.world.ALocFine)
        h1_product_operator = ProductOperator(h1_matrix, self.patch)

        super().__init__(
            operator,
            rhs,
            output_functional=output,
            products={"h1": h1_product_operator},
        )
        self.__auto_init(locals())

    def _assemble_for_aFine(self, aPatch):
        patch = self.patch
        world = self.world

        if aPatch.ndim == 1:
            ALocFine = world.ALocFine
        elif aPatch.ndim == 3:
            ALocFine = world.ALocMatrixFine

        iElementPatchCoarse = patch.iElementPatchCoarse
        elementFinetIndexMap = util.extractElementFine(
            patch.NPatchCoarse,
            world.NCoarseElement,
            iElementPatchCoarse,
            extractElements=True,
        )
        elementFinepIndexMap = util.extractElementFine(
            patch.NPatchCoarse,
            world.NCoarseElement,
            iElementPatchCoarse,
            extractElements=False,
        )

        AElementFull = fem.assemblePatchMatrix(world.NCoarseElement, ALocFine, aPatch[elementFinetIndexMap])

        ARhsList = list(patch.world.localBasis().T)
        self.numRhs = len(ARhsList)
        bPatchFullList = []
        for rhsIndex in range(self.numRhs):
            bPatchFull = np.zeros(patch.NpFine)
            bPatchFull[elementFinepIndexMap] += AElementFull * ARhsList[rhsIndex]
            bPatchFullList.append(bPatchFull)
        APatchFull = fem.assemblePatchMatrix(patch.NPatchFine, ALocFine, aPatch)

        return APatchFull, bPatchFullList

    def _compute_solution(self, mu=None, **kwargs):
        # NOTE : This function is only used in fom !
        aFinePatch = sum([aCoef(mu) * aPatch for (aPatch, aCoef) in zip(self.aFinePatches, self.aFineCoefficients)])

        IPatch = lambda: interp.L2ProjectionPatchMatrix(self.patch, self.world.boundaryConditions)
        # solver = lod.NullspaceOneLevelHierarchySolver(self.patch.NPatchCoarse, self.world.NCoarseElement)
        # solver = lod.DirectSolver()
        solver = None
        correctorsList = lod.computeBasisCorrectors(self.patch, IPatch, aFinePatch, saddleSolver=solver)

        c = correctorsList[self.dof]
        U = self.solution_space.make_array(c)
        return U


class CorrectorProblemOutput_for_a_DoF(Operator):
    linear = True

    def __init__(self, patch, aPatch, source):
        self.patch = patch
        self.source = source
        self.range = NumpyVectorSpace(patch.NpCoarse)  # only works for cubic mesh
        self.aPatch = aPatch

    def apply(self, U, mu=None):
        V = self.range.empty()
        for i in range(len(U)):
            u = U[i]
            correctorsList = [u.to_numpy().ravel()]
            csi = lod.computeSingleSeparatedBasisCoarseQuantities(self.patch, correctorsList, self.aPatch)
            V.append(self.range.make_array(csi.Qmsij.ravel()))
        return V