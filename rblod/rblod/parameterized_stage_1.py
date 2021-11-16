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
import scipy.sparse as sparse
from time import perf_counter

from gridlod import util, lod, interp, fem

from pymor.models.basic import StationaryModel
from pymor.operators.interface import Operator
from pymor.operators.constructions import VectorOperator, LincombOperator, ConstantOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.parameters.base import Parameters
from pymor.parameters.functionals import ProjectionParameterFunctional

from pymor.reductors.coercive import CoerciveRBReductor
from pymor.algorithms.greedy import rb_greedy
from rblod.optimized_rom import OptimizedNumpyModelStage1

"""
STAGE 1
"""

def build_two_scale_patch_models(patch, aFineCoefficients, coercivity_estimator,
                                 training_set_for_two_scale_stage_1, atol_patch, save_correctors,
                                 certified_estimator_study, aFine_Constructor,
                                 store=False, return_minimal=False, path=''):
    """
    prepare patch models with parameterized right hand side for TSRBLOD
    """
    tic = perf_counter()
    print(".", end="", flush=True)
    aPatch = aFine_Constructor(patch)
    m = CorrectorProblem_for_all_rhs(patch, aPatch, aFineCoefficients)
    reductor = CoerciveRBReductor(m, product=m.products["h1"], coercivity_estimator=coercivity_estimator)
    basis = reductor.bases["RB"] if save_correctors else None
    product = m.products["energy"] if certified_estimator_study else None
    h1_product = m.products["h1"] if certified_estimator_study else None
    greedy_data = rb_greedy(
        m, reductor, training_set=training_set_for_two_scale_stage_1,
        atol=atol_patch, extension_params={"method": "gram_schmidt"}
    )
    optimized_rom = OptimizedNumpyModelStage1(greedy_data["rom"])
    optimized_rom_minimal = optimized_rom.minimal_object()
    extension_failed = False if greedy_data["max_errs"][-1] < atol_patch else True
    source_dim = greedy_data["rom"].operator.source.dim
    max_errs = greedy_data["max_errs"]
    max_err_mus = greedy_data["max_err_mus"]
    K = m.Kij
    del optimized_rom, reductor, greedy_data, m, aPatch
    print("s", end="", flush=True)
    walltime = perf_counter() - tic
    if store:
        np.savez(f'{path}mpi_storage/ts_{patch.TInd}', rom=optimized_rom_minimal, K=K, time=walltime)
        return True
    elif return_minimal:
        return np.array([optimized_rom_minimal, K, walltime])
    else:
        return np.array([optimized_rom_minimal, walltime, source_dim, max_errs, max_err_mus, K,
                         extension_failed, basis, product, h1_product])


def correctors_from_TS_RBLOD_approach(optimized_rom, mu):
    """
    single online solve for parameterized version of stage 1
    Note: NOT used for the TSRBLOD, only used for reference if save_correctors is TRUE
    """
    u_roms = []
    for mu_ in _build_directional_mus(mu):
        u_rom = optimized_rom.solve(mu_)
        u_roms.append(u_rom.to_numpy())
    return u_roms


def _build_directional_mus(mu):
    canonical_directions = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    mu_with_canonical_directions = []
    parameters = {k: s for k, s in mu.parameters.items()}
    parameters["DoFs"] = 4
    parameters = Parameters(parameters)
    for e_i in canonical_directions:
        new_mu = {k: v for k, v in mu.items()}
        new_mu["DoFs"] = e_i
        mu_with_canonical_directions.append(parameters.parse(new_mu))
    return mu_with_canonical_directions

class CorrectorProblem_for_all_rhs(StationaryModel):
    def __init__(self, patch, aFinePatches, aFineCoefficients):
        self.patch = patch
        self.world = patch.world

        As, rhsss = zip(*(self._assemble_for_aFine(aFine) for aFine in aFinePatches))
        As_ = [NumpyMatrixOperator(A) for A in As]
        rhsss = [[As_[0].range.make_array(r) for r in rhss] for rhss in rhsss]
        operator = LincombOperator(As_, aFineCoefficients)

        rhs_coefficients = []
        rhs_operators = []
        pf_functionals = []
        for l in range(self.numRhs):
            pf = ProjectionParameterFunctional("DoFs", self.numRhs, l)
            pf_functionals.append(pf)
            rhs_coefficients.extend([c * pf for c in aFineCoefficients])
            for i in range(len(aFineCoefficients)):
                rhs = rhsss[i][l]
                rhs_op = VectorOperator(rhs)
                rhs_operators.append(rhs_op)
        rhs = LincombOperator(rhs_operators, rhs_coefficients)

        Kij_operators = []
        Kij_range = NumpyVectorSpace(4 * patch.NpCoarse)
        for aPatch in aFinePatches:
            csi = lod.computeSeparatedBasisCoarseQuantities(
                self.patch,
                [operator.source.zeros().to_numpy().ravel() for i in range(4)],
                aPatch,
            )
            Kij_operators.append(ConstantOperator(Kij_range.make_array(csi.Kmsij.ravel()), operator.source))
        self.Kij = LincombOperator(Kij_operators, aFineCoefficients)  # for the two scale matrix

        ### prepare output operator
        outer_operators = []
        outer_coefficients = []
        for aPatch in aFinePatches:
            outer_operators.append(CorrectorProblemOutput_for_a_DoF_globalized(patch, aPatch, operator.source))
        outer_coefficients.extend(aFineCoefficients)
        output = LincombOperator(outer_operators, outer_coefficients)

        fixed_aFinePatch = np.ones(aFinePatches[0].shape)
        energy_product = ProductOperator(self._assemble_for_aFine(fixed_aFinePatch)[0], self.patch)

        h1_matrix = fem.assemblePatchMatrix(patch.NPatchFine, patch.world.ALocFine)
        h1_product_operator = ProductOperator(h1_matrix, self.patch)

        super().__init__(
            operator,
            rhs,
            output_functional=output,
            products={"h1": h1_product_operator, "energy": energy_product},
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

        # solver = lod.DirectSolver()
        # solver = lod.NullspaceOneLevelHierarchySolver(self.patch.NPatchCoarse, self.world.NCoarseElement)
        solver = None
        correctorsList = lod.computeBasisCorrectors(self.patch, IPatch, aFinePatch, saddleSolver=solver)

        # extract DoF
        if np.sum(mu["DoFs"]) == 0:  # remove this after finishing the code since it is only for no mu[dofs] = 0
            DoF = [(0,)]
        else:
            assert np.sum(mu["DoFs"]) == 1
            DoF = np.where(mu["DoFs"] == 1)
        c = correctorsList[int(DoF[0])] * np.sum(mu["DoFs"])
        U = self.solution_space.make_array(c)
        return U


class CorrectorProblemOutput_for_a_DoF_globalized(Operator):
    linear = True

    def __init__(self, patch, aPatch, source):
        self.patch = patch
        self.source = source
        self.range = NumpyVectorSpace(patch.world.NpCoarse)  # only works for cubic mesh
        self.aPatch = aPatch
        # from local to global dofs
        patchpIndexMap = util.lowerLeftpIndexMap(patch.NPatchCoarse, patch.world.NWorldCoarse)
        patchpStartIndex = util.convertpCoordIndexToLinearIndex(patch.world.NWorldCoarse, patch.iPatchWorldCoarse)
        self.rows = patchpStartIndex + patchpIndexMap

    def apply(self, U, mu=None):
        V = self.range.empty()
        for i in range(len(U)):
            u = U[i]
            correctorsList = [u.to_numpy().ravel()]
            csi = lod.computeSingleSeparatedBasisCoarseQuantities(self.patch, correctorsList, self.aPatch)
            Qmsij_to_global = sparse.csc_matrix(
                (csi.Qmsij.ravel(), (self.rows, np.zeros_like(self.rows))),
                shape=(self.patch.world.NpCoarse, 1),
            )
            Qmsij_to_global_as_array = Qmsij_to_global.toarray().ravel()
            V.append(self.range.make_array(Qmsij_to_global_as_array))
        return V

class ProductOperator(NumpyMatrixOperator):
    #### FOR STAGE 1 #####
    def __init__(self, matrix, patch):
        self.matrix = matrix
        self.patch = patch
        self.world = patch.world
        super().__init__(matrix)

    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        patch = self.patch
        IPatch = interp.L2ProjectionPatchMatrix(self.patch, self.world.boundaryConditions)

        saddleSolver = lod.SchurComplementSolver()

        world = patch.world
        d = np.size(patch.NPatchCoarse)
        NPatchFine = patch.NPatchFine

        # Find what patch faces are common to the world faces, and inherit
        # boundary conditions from the world for those. For the other
        # faces, all DoFs fixed (Dirichlet)
        boundaryMapWorld = world.boundaryConditions == 0

        inherit0 = patch.iPatchWorldCoarse == 0
        inherit1 = (patch.iPatchWorldCoarse + patch.NPatchCoarse) == world.NWorldCoarse

        boundaryMap = np.ones([d, 2], dtype="bool")
        boundaryMap[inherit0, 0] = boundaryMapWorld[inherit0, 0]
        boundaryMap[inherit1, 1] = boundaryMapWorld[inherit1, 1]

        # Using schur complement solver for the case when there are no
        # Dirichlet conditions does not work. Fix if necessary.
        assert np.any(boundaryMap == True)

        fixed = util.boundarypIndexMap(NPatchFine, boundaryMap)
        V = V.to_numpy()
        U = saddleSolver.solve(self.matrix, IPatch, V, fixed, patch.NPatchCoarse, world.NCoarseElement)
        U = np.array(U)
        return self.source.from_numpy(U)
