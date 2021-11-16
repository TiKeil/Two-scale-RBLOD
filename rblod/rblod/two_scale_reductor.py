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

from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.operators.interface import Operator
from pymor.operators.constructions import IdentityOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.coercive import CoerciveRBEstimator
from pymor.reductors.residual import ResidualReductor
from pymor.reductors.basic import StationaryRBReductor
from pymor.vectorarrays.block import BlockVectorSpace

from rblod.two_scale_model import TwoScaleBlockOperator

"""
Two scale
"""

class CoerciveRBReductorForTwoScale(StationaryRBReductor):
    def __init__(
        self,
        world,
        optimized_romT,
        fom,
        RB=None,
        product=None,
        coercivity_estimator=None,
        check_orthonormality=None,
        check_tol=None,
        train_for="full"
    ):
        super().__init__(fom, RB, product, check_orthonormality, check_tol)
        self.__auto_init(locals())
        self.RBsizeT, self.residualT = zip(*(self._unpack_rom(rom_) for rom_ in optimized_romT))

        # res = a * coarse_part + b * fine_part
        self.residual_operator = self._construct_residual_operator(a=1, b=fom.rho_sqrt)
        self.residual_rhs = self._construct_residual_rhs(a=1)
        self.residual_product = self._construct_residual_product()

        self.full_residual_reductor = ResidualReductor(
            self.bases["RB"],
            self.residual_operator,
            self.residual_rhs,
            product=self.residual_product,
            riesz_representatives=True,
        )

        coarse_residual_operator = self._construct_residual_operator(a=1, b=0)
        coarse_residual_rhs = self._construct_residual_rhs(a=1)
        self.coarse_residual_reductor = ResidualReductor(
            self.bases["RB"],
            coarse_residual_operator,
            coarse_residual_rhs,
            product=self.residual_product,
            riesz_representatives=True,
        )

        fine_residual_operator = self._construct_residual_operator(a=0, b=fom.rho_sqrt)
        fine_residual_rhs = self._construct_residual_rhs(a=0)
        self.fine_residual_reductor = ResidualReductor(
            self.bases["RB"],
            fine_residual_operator,
            fine_residual_rhs,
            product=self.residual_product,
            riesz_representatives=True,
        )

        if train_for == 'full':
            self.residual_reductor = self.full_residual_reductor
        elif train_for == 'coarse':
            self.residual_reductor = self.coarse_residual_reductor
        else:
            assert 0, "this cannot happen"

    def _unpack_rom(self, rom):
        return rom.operator_array.shape[1], rom.error_residual

    def reduce(self):
        self._extract_partial_basis()
        return super().reduce()

    def _extract_partial_basis(self):
        partial_basis = self.bases["RB"].space.subspaces[0].empty()
        for basis in self.bases["RB"]:
            partial_basis.append(basis._blocks[0])
        self.partial_basis = partial_basis

    def reconstruct_partial(self, u):
        """Reconstruct first part of high-dimensional vector from blocked vector `u`."""
        return self.partial_basis.lincomb(u.to_numpy())

    def _construct_residual_operator(self, a, b):
        coefs = self.fom.operator.coefficients
        DTss = self._extract_D_matrices()
        operators = []
        for i, op in enumerate(self.fom.operator.operators):
            blocks = []
            DTs = DTss[i]
            if a == 1:
                blocks.append(op.blocks[0, :])
            else:
                blocks.append(op.blocks[0, :] * a)
            for j in range(len(self.RBsizeT)):
                new_line = [None for j_ in range(len(self.RBsizeT) + 1)]
                new_line[0] = b * DTs[j]
                new_line[j + 1] = b * self.residualT[j].operator.operators[i]
                blocks.append(new_line)
            operators.append(TwoScaleBlockOperator(np.array(blocks)))
        return LincombOperator(operators, coefs)

    def _construct_residual_rhs(self, a):
        rhs_space = self.fom.operator.operators[0].blocks[0, 0].source
        if a == 1:
            coarse_rhs = VectorOperator(rhs_space.from_numpy(self.fom.bFree))
        else:
            coarse_rhs = VectorOperator(rhs_space.from_numpy(self.fom.bFree) * a)
        blocked_rhs_length = len(self.residual_operator.range.zeros().to_numpy()[0])
        rhs_as_array = coarse_rhs.as_vector().to_numpy()[0]
        block_rhs = [rhs for rhs in rhs_as_array]
        for i in range(blocked_rhs_length - len(rhs_as_array)):
            block_rhs.append(0)
        block_rhs = np.array(block_rhs).flatten()
        blocked_rhs = VectorOperator(self.residual_operator.range.from_numpy(block_rhs))
        return blocked_rhs

    def _construct_residual_product(self):
        blocks = [self.product.blocks[0]]
        blocks.extend([IdentityOperator(sp) for sp in self.residual_operator.operators[0].range_spaces[1:]])
        return TrueDiagonalBlockOperator(blocks, True)

    def _extract_D_matrices(self):
        DTss = []
        for i in range(self.fom.affine_components):
            DTs = []
            for j, residual in enumerate(self.residualT):
                rhs = residual.rhs
                TPrimes = self.fom.TPrimeCoarsepStartIndices[j] + self.fom.TPrimeCoarsepIndexMap
                Dij = np.zeros((self.fom.NpCoarse, rhs.range.dim))
                for l in range(4):
                    TPrime = TPrimes[l]
                    Dij[TPrime] = rhs.operators[self.fom.affine_components * l + i].matrix.T[0]
                DTs.append(-NumpyMatrixOperator(sparse.csr_matrix(Dij[self.fom.free])).H)
            DTss.append(DTs)
        return DTss

    def assemble_error_estimator(self):
        residual = self.residual_reductor.reduce()
        error_estimator = CoerciveRBEstimator(residual, tuple(self.residual_reductor.residual_range_dims),
                                        self.coercivity_estimator)
        return error_estimator

    def assemble_full_error_estimator(self):
        residual = self.full_residual_reductor.reduce()
        error_estimator = CoerciveRBEstimator(residual, tuple(self.full_residual_reductor.residual_range_dims),
                                        self.coercivity_estimator)
        return error_estimator

    def assemble_coarse_error_estimator(self):
        residual = self.coarse_residual_reductor.reduce()
        error_estimator = CoerciveRBEstimator(residual, tuple(self.coarse_residual_reductor.residual_range_dims),
                                        self.coercivity_estimator)
        return error_estimator

    def assemble_fine_error_estimator(self):
        residual = self.fine_residual_reductor.reduce()
        error_estimator = CoerciveRBEstimator(residual, tuple(self.fine_residual_reductor.residual_range_dims),
                                        self.coercivity_estimator)
        return error_estimator

    def assemble_error_estimator_for_subbasis(self, dims):
        return NotImplemented

class TrueDiagonalBlockOperator(Operator):
    def _operators(self):
        return NotImplementedError

    def __init__(self, blocks, only_first=False):
        self.only_first = only_first
        self.blocked_source = True
        self.blocked_range = True
        self.blocks = blocks = np.array(blocks)
        assert 1 <= blocks.ndim <= 2

        # find source/range spaces for every column/row
        self.source_spaces = [None for j in range(len(blocks))]
        self.range_spaces = [None for i in range(len(blocks))]
        for i, op in enumerate(blocks):
            if op is not None:
                assert self.source_spaces[i] is None or op.source == self.source_spaces[i]
                self.source_spaces[i] = op.source
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
        # we now that the matrix has the sparse structure
        # A
        #   C_T
        #       C_T
        #           C_T
        #               C_T
        if self.only_first:
            V_blocks = [self.blocks[0].apply(U.block(0), mu=mu)]
            V_blocks.extend(U._blocks[1:])
        else:
            V_blocks = [self.blocks[i].apply(U.block(i), mu=mu) for i in range(self.num_range_blocks)]
        return self.range.make_array(V_blocks)

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
        assert V in self.range
        assert initial_guess is None or initial_guess in self.source and len(initial_guess) == len(V)
        if self.only_first:
            U_blocks = [self.blocks[0].apply_inverse(V.block(0), mu=mu,
                                                     initial_guess=(initial_guess.block(0)
                                                                    if initial_guess is not None else None),
                                                     least_squares=least_squares)]
            U_blocks.extend(V._blocks[1:])
        else:
            U_blocks = [self.blocks[i].apply_inverse(V.block(i), mu=mu,
                                                     initial_guess=(initial_guess.block(i)
                                                                    if initial_guess is not None else None),
                                                     least_squares=least_squares)
                        for i in range(self.num_source_blocks)]
        return self.source.make_array(U_blocks)