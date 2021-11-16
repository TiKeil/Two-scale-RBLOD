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

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.models.basic import StationaryModel

from pymor.operators.constructions import VectorOperator, LincombOperator

from gridlod import pglod, util, fem

from rblod.parameterized_stage_1 import _build_directional_mus

from pymor.operators.interface import Operator
from pymor.vectorarrays.block import BlockVectorSpace


class TwoScaleBlockOperatorBase(Operator):
    def _operators(self):
        """Iterator over operators."""
        for (i, j) in np.ndindex(self.blocks.shape):
            yield self.blocks[i, j]

    def __init__(self, blocks):
        self.blocks = blocks = np.array(blocks)
        assert 1 <= blocks.ndim <= 2
        if self.blocked_source and self.blocked_range:
            assert blocks.ndim == 2
        elif self.blocked_source:
            if blocks.ndim == 1:
                blocks.shape = (1, len(blocks))
        else:
            if blocks.ndim == 1:
                blocks.shape = (len(blocks), 1)

        # find source/range spaces for every column/row
        self.source_spaces = [None for j in range(blocks.shape[1])]
        self.range_spaces = [None for i in range(blocks.shape[0])]
        for (i, j), op in np.ndenumerate(blocks):
            if op is not None:
                assert self.source_spaces[j] is None or op.source == self.source_spaces[j]
                self.source_spaces[j] = op.source
                assert self.range_spaces[i] is None or op.range == self.range_spaces[i]
                self.range_spaces[i] = op.range

        self.source = BlockVectorSpace(self.source_spaces) if self.blocked_source else self.source_spaces[0]
        self.range = BlockVectorSpace(self.range_spaces) if self.blocked_range else self.range_spaces[0]
        self.num_source_blocks = len(self.source_spaces)
        self.num_range_blocks = len(self.range_spaces)
        self.linear = True

    @property
    def H(self):
        return NotImplementedError

    def apply(self, U, mu=None):
        assert U in self.source
        # we know that the matrix has the sparse structure
        # A    B_T  B_T ... B_T
        # D_T  C_T
        # D_T       C_T
        # .             C_T
        # D_T               C_T
        # thus we can reduce the computational cost by only iterating over these matrices leaving the rest untouched.
        V_blocks = [None for i in range(self.num_range_blocks)]
        # first row
        V_blocks[0] = self.blocks[0][0].apply(U.block(0))
        for i, op in enumerate(self.blocks[0][1:], 1):
            V_blocks[0] += op.apply(U.block(i))
        # all other rows
        for i, block in enumerate(self.blocks[1:], 1):
            V_blocks[i] = block[0].apply(U.block(0))
            V_blocks[i] += block[i].apply(U.block(i))

        return self.range.make_array(V_blocks) if self.blocked_range else V_blocks[0]

    def apply_adjoint(self, V, mu=None):
        return NotImplementedError

    def assemble(self, mu=None):
        return NotImplementedError

    def as_range_array(self, mu=None):
        return NotImplementedError

    def as_source_array(self, mu=None):
        return NotImplementedError

    def d_mu(self, parameter, index=0):
        return NotImplementedError

    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        return NotImplementedError


class TwoScaleBlockOperator(TwoScaleBlockOperatorBase):
    """A matrix of arbitrary |Operators|.

    This operator can be :meth:`applied <pymor.operators.interface.Operator.apply>`
    to a compatible :class:`BlockVectorArrays <pymor.vectorarrays.block.BlockVectorArray>`.

    Parameters
    ----------
    blocks
        Two-dimensional array-like where each entry is an |Operator| or `None`.
    """

    blocked_source = True
    blocked_range = True


"""
STAGE 2
"""


class Two_Scale_Problem(StationaryModel):
    # for the two scale matrix
    def __init__(self, optimized_romT, KijT, f, patchT, contrast, error_estimator=None, name=None):
        self.__auto_init(locals())
        opT, rhsT, outputT = zip(*(self._unpack_rom(rom) for rom in optimized_romT))
        self.world = patchT[0].world
        self.NpCoarse = np.prod(self.world.NWorldCoarse + 1)
        aFineCoefficients = optimized_romT[0].op_coefficients
        self.affine_components = len(aFineCoefficients)

        # for handling dirichlet dofs
        self.free = util.interiorpIndexMap(self.world.NWorldCoarse)

        # dof handling of the global coarse matrix
        NPatchCoarse = self.world.NWorldCoarse
        self.TPrimeCoarsepStartIndices = util.lowerLeftpIndexMap(NPatchCoarse - 1, NPatchCoarse)
        self.TPrimeCoarsepIndexMap = util.lowerLeftpIndexMap(np.ones_like(NPatchCoarse), NPatchCoarse)

        # compute weight rho
        C_ovl = (2*patchT[0].k + 1)**2     # for quadrilateral grids we have (2k + 1)**2 patches per element
        self.rho_sqrt = np.sqrt(C_ovl * contrast)

        # A    B_T  B_T ... B_T
        # D_T  C_T
        # D_T       C_T
        # .             C_T
        # D_T               C_T

        As = self._extract_A_matrices()
        BTss = self._extract_B_matrices(outputT)
        CTss = self._extract_C_matrices(opT)
        DTss = self._extract_D_matrices(rhsT)

        operators = []
        for A, BTs, CTs, DTs in zip(As, BTss, CTss, DTss):
            blocks = []
            first_line = [A]
            first_line.extend(BTs)
            blocks.append(first_line)
            for i, (BT, DT, CT) in enumerate(zip(BTs, DTs, CTs)):
                new_line = [DT]
                for k in range(len(BTs)):
                    if k == i:
                        new_line.append(CT)
                    else:
                        new_line.append(None)
                blocks.append(new_line)
            operators.append(TwoScaleBlockOperator(blocks=np.array(blocks)))

        operator = LincombOperator(operators, aFineCoefficients)
        rhs_space = operator.operators[0].blocks[0][0].source
        MFull = fem.assemblePatchMatrix(self.world.NWorldCoarse, self.world.MLocCoarse)
        bFull = MFull * f
        self.bFree = bFull[self.free]
        coarse_rhs = VectorOperator(rhs_space.from_numpy(self.bFree))
        blocked_rhs_length = len(operator.source.zeros().to_numpy()[0])

        rhs_as_array = coarse_rhs.as_vector().to_numpy()[0]
        block_rhs = [rhs for rhs in rhs_as_array]
        for i in range(blocked_rhs_length - len(rhs_as_array)):
            block_rhs.append(0)
        block_rhs = np.array(block_rhs).flatten()
        blocked_rhs = VectorOperator(operator.source.from_numpy(block_rhs))

        super().__init__(operator, blocked_rhs, error_estimator=error_estimator, name=name)
        self.parameters_own = operator.parameters
        self.parameters = operator.parameters

    def _unpack_rom(self, rom):
        return rom.operator_array, rom.rhs_array, rom.output_array

    def _extract_A_matrices(self):
        As = []
        for i in range(self.affine_components):
            Kij = []
            for output in self.KijT:
                Kij.append(output.operators[i].value.to_numpy())
            A = pglod.assembleMsStiffnessMatrix(self.world, self.patchT, Kij)
            A_free = A[self.free][:, self.free]
            As.append(NumpyMatrixOperator(sparse.csr_matrix(A_free)))
        return As

    def _extract_B_matrices(self, outputT):
        BTss = []
        for i in range(self.affine_components):
            BTs = []
            for output in outputT:
                free_matrix = output[i][self.free]
                free_matrix_sparse = sparse.csr_matrix(free_matrix)
                free_operator = NumpyMatrixOperator(free_matrix_sparse)
                BTs.append(free_operator)
            BTss.append(BTs)
        return BTss

    def _extract_C_matrices(self, opT):
        CTss = []
        for i in range(self.affine_components):
            CTs = []
            for op in opT:
                CTs.append(self.rho_sqrt * NumpyMatrixOperator(op[i]))
            CTss.append(CTs)
        return CTss

    def _extract_D_matrices(self, rhsT):
        DTss = []
        for i in range(self.affine_components):
            DTs = []
            for j, rhs in enumerate(rhsT):
                TPrimes = self.TPrimeCoarsepStartIndices[j] + self.TPrimeCoarsepIndexMap
                Dij = np.zeros((self.NpCoarse, rhs.shape[1]))
                for l in range(4):
                    TPrime = TPrimes[l]
                    rhs_col = rhs[self.affine_components * l + i].T[0]
                    Dij[TPrime] = rhs_col
                Dij_sparse_free = sparse.csr_matrix(Dij[self.free])
                DTs.append(-NumpyMatrixOperator(Dij_sparse_free).H * self.rho_sqrt)
            DTss.append(DTs)
        return DTss

    def _compute_RBLOD_correctors(self, mu, TInd):
        mu_with_canonical_directions = _build_directional_mus(mu)
        Kmsij = []
        ################################
        for mu_ in mu_with_canonical_directions:
            DoF = np.where(mu_["DoFs"] == 1)
            Qmsij = []
            output = self.optimized_romT[TInd].output(mu_)[0]
            output = output[output != 0]
            for i in range(4):
                Qmsij.append(list((i == DoF[0]) * output))
            Kmsij.append(np.column_stack(Qmsij).flatten())
        Kmsij.append(self.KijT[TInd].apply(self.KijT[TInd].source.zeros(), mu).to_numpy()[0])
        ################################
        Kmsij_from_rom = np.einsum("ti->i", Kmsij)
        ################################
        return Kmsij_from_rom

    def _loop_over_T(self, mu, u_H, TInd):
        TPrimes = self.TPrimeCoarsepStartIndices[TInd] + self.TPrimeCoarsepIndexMap
        mu_ = {k: v for k, v in mu.items()}
        mu_["DoFs"] = u_H[TPrimes]
        u_f = self.optimized_romT[TInd].solve(self.optimized_romT[TInd].parameters.parse(mu_))
        return list(u_f.to_numpy()[0])

    def _compute_solution(self, mu=None, **kwargs):
        # for the FOM we do not want to solve the Block system because it becomes to expensive
        # thus we split the computations

        # compute the old ms stiffness matrix and compute u_H with it.
        KmsijT = []
        for TInd in range(self.world.NtCoarse):
            # not parallel since it is only in the offline phase
            KmsijT.append(self._compute_RBLOD_correctors(mu, TInd))
        KFull_old_RB = pglod.assembleMsStiffnessMatrix(self.world, self.patchT, KmsijT)
        KFree_old_RB = KFull_old_RB[self.free][:, self.free]
        u_H = sparse.linalg.spsolve(KFree_old_RB, self.bFree)

        # now, fill the block vector with the help of u_H
        xFull_old_RB = np.zeros(self.world.NpCoarse)
        xFull_old_RB[self.free] = u_H
        u = np.array(u_H)
        for TInd in range(self.world.NtCoarse):
            u = np.append(u, self._loop_over_T(mu, xFull_old_RB, TInd))
        return self.solution_space.from_numpy(u)
