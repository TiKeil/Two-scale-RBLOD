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
import time
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.models.basic import StationaryModel

from pymor.operators.constructions import VectorOperator, LincombOperator

from gridlod import pglod, util, fem

from rblod.parameterized_stage_1 import _build_directional_mus

from pymor.operators.interface import Operator
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace

class EfficientTwoScaleBlockOperator(Operator):
    def _operators(self):
        raise NotImplementedError

    def __init__(self, A, BT, CT, DT, source_spaces, range_spaces):
        self.source_spaces = source_spaces
        self.range_spaces = range_spaces
        self.A = A
        self.BT = BT
        self.CT = CT
        self.DT = DT

        self.blocked_source = True
        self.blocked_range = True
        self.source = BlockVectorSpace(self.source_spaces)
        self.range = BlockVectorSpace(self.range_spaces)
        self.num_source_blocks = len(self.source_spaces)
        self.num_range_blocks = len(self.range_spaces)
        self.linear = True

        self.A_mat = A.matrix
        self.B_mats = [B.matrix if B else None for B in BT]
        self.C_mats = [C.operators[0].matrix * C.coefficients[0] if C else None for C in CT]
        self.D_mats = [D.operators[0].matrix * D.coefficients[0] if D else None for D in DT]

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
        U_block_0 = U.block(0).to_numpy()
        V_blocks[0] = self.A_mat.dot(U_block_0.T).T
        for i, (B, C, D) in enumerate(zip(self.B_mats, self.C_mats, self.D_mats), 1):
            if C is not None:
                U_block = U.block(i).to_numpy()
                # first row
                V_blocks[0] += B.dot(U_block.T).T
                # all other rows
                V_blocks[i] = D.dot(U_block_0.T).T
                V_blocks[i] += C.dot(U_block.T).T
            else:
                V_blocks[i] = self.source_spaces[i].zeros(len(U)).to_numpy()
        V_blocks = np.concatenate(V_blocks, axis=None).ravel()
        V = self.range.from_numpy(V_blocks)
        # print(V.to_numpy()[0])
        return self.range.from_numpy(V_blocks)

    def apply_adjoint(self, V, mu=None):
        raise NotImplementedError

    def assemble(self, mu=None):
        raise NotImplementedError

    def as_range_array(self, mu=None):
        raise NotImplementedError

    def as_source_array(self, mu=None):
        raise NotImplementedError

    def d_mu(self, parameter, index=0):
        raise NotImplementedError

    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        raise NotImplementedError


class TwoScaleBlockOperator(Operator):
    def _operators(self):
        raise NotImplementedError

    def __init__(self, A, BT, CT, DT, source_spaces, range_spaces, a=1, b=1):
        self.source_spaces = source_spaces
        self.range_spaces = range_spaces
        self.A = A
        self.BT = BT
        self.CT = CT
        self.DT = DT
        self.b = b
        self.a = a

        self.blocked_source = True
        self.blocked_range = True
        self.source = BlockVectorSpace(self.source_spaces)
        self.range = BlockVectorSpace(self.range_spaces)
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
        V_blocks[0] = self.A.apply(U.block(0))
        for i, (B, C, D) in enumerate(zip(self.BT, self.CT, self.DT), 1):
            if B is not None:
                assert C is not None and D is not None
                U_block = U.block(i)
                # first row
                V_blocks[0] += B.apply(U_block)
                V_blocks[i] = D.apply(U.block(0))
                V_blocks[i] += C.apply(U_block)
            else:
                V_blocks[i] = self.range_spaces[i].zeros(len(U))
        V = self.range.make_array(V_blocks)
        # print(V.to_numpy()[0])
        return self.range.make_array(V_blocks) if self.blocked_range else V_blocks[0]

    def apply_adjoint(self, V, mu=None):
        raise NotImplementedError

    def assemble(self, mu=None):
        # for this you need to assemble the None parts !
        assert not self.can_not_be_assembled
        blocks = np.empty(self.blocks.shape, dtype=object)
        for (i, j) in np.ndindex(self.blocks.shape):
            blocks[i, j] = self.blocks[i, j].assemble(mu)
        if np.all(blocks == self.blocks):
            return self
        else:
            return self.__class__(blocks)

    def as_range_array(self, mu=None):
        raise NotImplementedError

    def as_source_array(self, mu=None):
        raise NotImplementedError

    def d_mu(self, parameter, index=0):
        raise NotImplementedError

    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        raise NotImplementedError


class SimplifiedBlockColumnOperator(Operator):
    def _operators(self):
        return NotImplementedError

    def __init__(self, blocks, range_spaces=None, range=None):
        self.only_first = True
        self.blocked_source = False
        self.blocked_range = True
        self.blocks = blocks = np.array(blocks)
        assert 1 <= blocks.ndim <= 2

        # find source/range spaces for every column/row
        self.source_spaces = [NumpyVectorSpace(1)]
        self.range_spaces = range_spaces

        self.source = BlockVectorSpace(self.source_spaces) if self.blocked_source else self.source_spaces[0]
        self.num_source_blocks = len(self.source_spaces)

        if range is None:
            self.range = BlockVectorSpace(self.range_spaces) if self.blocked_range else self.range_spaces[0]
            self.num_range_blocks = len(self.range_spaces)
        else:
            self.range = range
            self.num_range_blocks = len(range.subspaces)
        self.linear = True

    @property
    def H(self):
        raise NotImplementedError

    def apply(self, U, mu=None):
        return NotImplementedError

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        assert self.only_first
        return self.blocks[0].apply_adjoint(V.block(0), mu=mu)

    def assemble(self, mu=None):
        raise NotImplementedError

    def as_range_array(self, mu=None):
        assert self.only_first
        def process_op(op, space):
            R = space.empty()
            R.append(op.as_range_array(mu))
            return R

        subspaces = self.range.subspaces if self.blocked_range else [self.range]
        blocks = [process_op(self.blocks[0], subspaces[0])]
        blocks.extend([space.zeros() for space in subspaces[1:]])
        return self.range.make_array(blocks) if self.blocked_range else blocks[0]

    def as_source_array(self, mu=None):
        raise NotImplementedError

    def d_mu(self, parameter, index=0):
        raise NotImplementedError

"""
STAGE 2
"""
from pymor.parameters.functionals import ProjectionParameterFunctional

def is_equal(first, second):
    if isinstance(first, float) and isinstance(second, float):
        return first == second
    if isinstance(first, ProjectionParameterFunctional) and isinstance(second, ProjectionParameterFunctional):
        if first.parameter == second.parameter and first.index == second.index:
            return True
    # TODO: add more cases !!
    return False

class Two_Scale_Problem(StationaryModel):
    # for the two scale matrix
    def __init__(self, optimized_romT, KijT, f, patchT, aFineCoefficients, contrast, error_estimator=None, name=None,
                 As=None, BTss=None, CTss=None, DTss=None, source_spaces=None):
        self.__auto_init(locals())
        opT, rhsT, outputT, coefsT = zip(*(self._unpack_rom(rom) for rom in optimized_romT))
        self.world = patchT[0].world
        self.NpCoarse = np.prod(self.world.NWorldCoarse + 1)
        self.NCoarse = np.prod(self.world.NWorldCoarse)
        self.affine_components = len(aFineCoefficients)

        # it is not possible to have the same coefficients two times in the list
        for coef in aFineCoefficients:
            equal = 0
            for coef_ in aFineCoefficients:
                if is_equal(coef, coef_):
                    equal += 1
            assert equal == 1

        # for handling dirichlet dofs
        self.free = util.interiorpIndexMap(self.world.NWorldCoarse)

        # dof handling of the global coarse matrix
        NPatchCoarse = self.world.NWorldCoarse
        self.TPrimeCoarsepStartIndices = util.lowerLeftpIndexMap(NPatchCoarse - 1, NPatchCoarse)
        self.TPrimeCoarsepIndexMap = util.lowerLeftpIndexMap(np.ones_like(NPatchCoarse), NPatchCoarse)

        # compute weight rho
        C_ovl = (2 * patchT[0].k + 1) ** 2
        self.rho_sqrt = np.sqrt(C_ovl * contrast)

        # A    B_T  B_T ... B_T
        # D_T  C_T
        # D_T       C_T
        # .             C_T
        # D_T               C_T

        if As is None:
            tic_ = time.perf_counter()
            print('construct A ...' , end='', flush=True)
            self.As = self._extract_A_matrices()
            print(f' in {time.perf_counter()-tic_:.5f}s')
        if BTss is None:
            tic_ = time.perf_counter()
            print('construct B ...' , end='', flush=True)
            self.BTss, self.source_spaces = self._extract_B_matrices(outputT, coefsT)
            print(f' in {time.perf_counter()-tic_:.5f}s')
        if CTss is None:
            tic_ = time.perf_counter()
            print('construct C ...' , end='', flush=True)
            self.CTss = self._extract_C_matrices(opT, coefsT)
            print(f' in {time.perf_counter()-tic_:.5f}s')
        if DTss is None:
            tic_ = time.perf_counter()
            print('construct D ...' , end='', flush=True)
            self.DTss = self._extract_D_matrices(rhsT, coefsT)
            print(f' in {time.perf_counter()-tic_:.5f}s')

        tic_ = time.perf_counter()
        print('constructing block operator ...' , end='', flush=True)
        operators = []
        # find source/range spaces for every column/row before block operator
        range_spaces = [self.As[0].source]
        range_spaces.extend(self.source_spaces)
        source_spaces = range_spaces

        for A, BTs, CTs, DTs in zip(self.As, self.BTss, self.CTss, self.DTss):
            operators.append(TwoScaleBlockOperator(A, BTs, CTs, DTs,
                                                   source_spaces=source_spaces,
                                                   range_spaces=range_spaces))

        # # TODO: this makes Stage 2 fast but does not work yet
        # for A, BTs, CTs, DTs in zip(self.As, self.BTss, self.CTss, self.DTss):
        #     operators.append(EfficientTwoScaleBlockOperator(A, BTs, CTs, DTs,
        #                                            source_spaces=source_spaces,
        #                                            range_spaces=range_spaces))
        print(f' in {time.perf_counter()-tic_:.5f}s')

        operator = LincombOperator(operators, aFineCoefficients)

        #rhs
        rhs_space = A.source

        if isinstance(f, np.ndarray):
            # then f is defined in a gridlod way
            self.MCoarse = fem.assemblePatchMatrix(self.world.NWorldCoarse, self.world.MLocCoarse)
            bCoarseFull = self.MCoarse * f
            bCoarseFull_ = NumpyVectorSpace(len(bCoarseFull)).from_numpy(bCoarseFull)
            self.bCoarseFull_op = VectorOperator(bCoarseFull_)
        else:
            # then f is defined in a pymor where MCoarse is already incorporated by the descritization
            assert isinstance(f, VectorOperator) or isinstance(f, LincombOperator)
            self.bCoarseFull_op = f

        if isinstance(self.bCoarseFull_op, VectorOperator):
            bFull = self.bCoarseFull_op.as_vector().to_numpy()[0]
            bFree = bFull[self.free]
            coarse_rhs = VectorOperator(rhs_space.from_numpy(bFree))

            # blocked rhs
            block_rhs = [coarse_rhs]
            blocked_rhs = SimplifiedBlockColumnOperator(block_rhs, range_spaces=range_spaces)
        else:
            assert isinstance(self.bCoarseFull_op, LincombOperator)
            rhs_ops = []
            for f_op in self.bCoarseFull_op.operators:
                bFull = f_op.as_vector().to_numpy()[0]
                bFree = bFull[self.free]
                rhs_ops.append(VectorOperator(rhs_space.from_numpy(bFree)))

            blocks = []
            # make this affinely decomposable
            for c_op in rhs_ops:
                # blocked rhs
                block_rhs = [c_op]
                blocks.append(SimplifiedBlockColumnOperator(block_rhs, range_spaces=range_spaces))
            blocked_rhs = LincombOperator(blocks, self.bCoarseFull_op.coefficients)

        super().__init__(operator, blocked_rhs, error_estimator=error_estimator, name=name)

    def extract_ABCD_and_sp(self):
        return self.As, self.BTss, self.CTss, self.DTss, self.source_spaces

    def _unpack_rom(self, rom):
        return rom.operator_array, rom.rhs_array, rom.output_array, rom.output_coefficients

    def _extract_A_matrices(self):
        As = []
        for coef in self.aFineCoefficients:
            Kij = []
            for output in self.KijT:
                success = 0
                for coef_, op in zip(output.coefficients, output.operators):
                    if is_equal(coef_, coef):
                        Kij.append(op.value.to_numpy())
                        success = 1
                        break
                if not success:
                    Kij.append(op.range.zeros().to_numpy())
            A = pglod.assembleMsStiffnessMatrix(self.world, self.patchT, Kij)
            A_free = A[self.free][:, self.free]
            As.append(NumpyMatrixOperator(sparse.csr_matrix(A_free)))
        return As

    def _extract_B_matrices(self, outputT, coefsT):
        BTss = []
        source_spaces = [None for _ in range(len(coefsT))]
        for coef in self.aFineCoefficients:
            BTs = []
            for i, (output, coefs) in enumerate(zip(outputT, coefsT)):
                success = 0
                for coef_, out in zip(coefs, output):
                    if is_equal(coef_, coef):
                        free_matrix = out[self.free]
                        success = 1
                        free_matrix_sparse = sparse.csr_matrix(free_matrix)
                        free_operator = NumpyMatrixOperator(free_matrix_sparse)
                        BTs.append(free_operator)
                        if source_spaces[i] is None:
                            source_spaces[i] = free_operator.source
                        break
                if not success:
                    BTs.append(None)
            BTss.append(BTs)
        return BTss, source_spaces

    def _extract_C_matrices(self, opT, coefsT):
        CTss = []
        for coef in self.aFineCoefficients:
            CTs = []
            for ops, coefs in zip(opT, coefsT):
                success = 0
                for coef_, op in zip(coefs, ops):
                    if is_equal(coef_, coef):
                        CTs.append(self.rho_sqrt * NumpyMatrixOperator(op))
                        success = 1
                        break
                if not success:
                    CTs.append(None)
            CTss.append(CTs)
        return CTss

    def _extract_D_matrices(self, rhsT, coefsT):
        DTss = []
        for coef in self.aFineCoefficients:
            DTs = []
            for j, (rhss, coefs) in enumerate(zip(rhsT, coefsT)):
                success = 0
                TPrimes = self.TPrimeCoarsepStartIndices[j] + self.TPrimeCoarsepIndexMap
                shape = rhss.shape[1]
                rhs_reshaped = rhss.reshape(len(coefs), 4, shape, 1, order='F')
                for coef_, rhs in zip(coefs, rhs_reshaped):
                    if is_equal(coef_, coef):
                        Dij = np.zeros((self.NpCoarse, shape))
                        for l in range(4):
                            TPrime = TPrimes[l]
                            rhs_col = rhs[l].T[0]
                            Dij[TPrime] = rhs_col
                        success = 1
                        break
                if success:
                    Dij_sparse_free = sparse.csr_matrix(Dij[self.free])
                    DTs.append(-NumpyMatrixOperator(Dij_sparse_free).H * self.rho_sqrt)
                else:
                    DTs.append(None)
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
        mu_ = {k: v for k, v in mu.items() if k != "basis_coefficients"}
        mu_["DoFs"] = u_H[TPrimes]
        u_f = self.optimized_romT[TInd].solve(self.optimized_romT[TInd].parameters.parse(mu_))
        return list(u_f.to_numpy()[0])

    def _solve(self, mu=None, **kwargs):
        # for the FOM we do not want to solve the block system because it becomes to expensive
        # thus we split the computations

        # compute the old ms stiffness matrix and compute u_H with it.
        KmsijT = []
        for TInd in range(self.world.NtCoarse):
            # not parallel since it is only in the offline phase
            KmsijT.append(self._compute_RBLOD_correctors(mu, TInd))
        KFull_old_RB = pglod.assembleMsStiffnessMatrix(self.world, self.patchT, KmsijT)
        KFree_old_RB = KFull_old_RB[self.free][:, self.free]

        bFull = self.bCoarseFull_op.as_vector(mu).to_numpy()[0]
        bFree = bFull[self.free]
        u_H = sparse.linalg.spsolve(KFree_old_RB, bFree)

        # now, fill the block vector with the help of u_H
        xFull_old_RB = np.zeros(self.world.NpCoarse)
        xFull_old_RB[self.free] = u_H
        u = np.array(u_H)
        for TInd in range(self.world.NtCoarse):
            u = np.append(u, self._loop_over_T(mu, xFull_old_RB, TInd))
        return self.solution_space.from_numpy(u)