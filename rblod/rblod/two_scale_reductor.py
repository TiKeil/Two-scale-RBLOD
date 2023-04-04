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

from pymor.operators.constructions import LincombOperator
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.coercive import CoerciveRBEstimator
from pymor.reductors.residual import ResidualReductor
from pymor.reductors.basic import StationaryRBReductor
from pymor.vectorarrays.block import BlockVectorSpace

from rblod.two_scale_model import TwoScaleBlockOperator, is_equal, SimplifiedBlockColumnOperator

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
        train_for="full",
        residual_operator=None,
        residual_product=None
    ):
        super().__init__(fom, RB, product, check_orthonormality, check_tol)
        self.__auto_init(locals())
        self.RBsizeT, self.residualT = zip(*(self._unpack_rom(rom_) for rom_ in optimized_romT))

        if self.residualT[0] is not None: # then we do not have reductor_type = non_assembled
            # res = a * coarse_part + b * fine_part
            if residual_operator is None:
                self.residual_operator = self._construct_residual_operator(a=1, b=fom.rho_sqrt)
            self.residual_rhs = self._construct_residual_rhs(a=1)
            if residual_product is None:
                self.residual_product = self._construct_residual_product()

            self.full_residual_reductor = ResidualReductor(
                self.bases["RB"],
                self.residual_operator,
                self.residual_rhs,
                product=self.residual_product,
                riesz_representatives=True,
            )

            # coarse_residual_operator = self._construct_residual_operator(a=1, b=0)
            # coarse_residual_rhs = self._construct_residual_rhs(a=1)
            # self.coarse_residual_reductor = ResidualReductor(
            #     self.bases["RB"],
            #     coarse_residual_operator,
            #     coarse_residual_rhs,
            #     product=self.residual_product,
            #     riesz_representatives=True,
            # )
            #
            # fine_residual_operator = self._construct_residual_operator(a=0, b=fom.rho_sqrt)
            # fine_residual_rhs = self._construct_residual_rhs(a=0)
            # self.fine_residual_reductor = ResidualReductor(
            #     self.bases["RB"],
            #     fine_residual_operator,
            #     fine_residual_rhs,
            #     product=self.residual_product,
            #     riesz_representatives=True,
            # )

            if train_for == 'full':
                self.residual_reductor = self.full_residual_reductor
            elif train_for == 'coarse':
                self.residual_reductor = self.coarse_residual_reductor
            else:
                assert 0, "this cannot happen"
        else:
            self.residual_reductor = None

    def _unpack_rom(self, rom):
        if rom.rom is None:
            return rom.operator_array.shape[1], rom.error_residual
        else:
            return rom.operator_array.shape[1], rom.rom.error_estimator.residual

    def extract_op_prod(self):
        return self.residual_operator, self.residual_product

    def reduce(self):
        self._extract_partial_basis()
        print('reducing reductor')
        return super().reduce()

    def _extract_partial_basis(self):
        partial_basis = self.bases["RB"].space.subspaces[0].empty()
        for basis in self.bases["RB"]:
            partial_basis.append(basis.blocks[0])
        self.partial_basis = partial_basis

    def reconstruct_partial(self, u):
        """Reconstruct first part of high-dimensional vector from blocked vector `u`."""
        return self.partial_basis.lincomb(u.to_numpy())

    def _construct_residual_operator(self, a, b):
        coefs = self.fom.operator.coefficients
        DTss, new_range_spaces = self._extract_D_matrices(coefs)
        range_spaces = [self.fom.operator.operators[0].A.source]
        range_spaces.extend(new_range_spaces)
        source_spaces = [self.fom.operator.operators[0].A.source]
        source_spaces.extend(self.fom.source_spaces)
        operators = []
        for i, (op, coef, DTs) in enumerate(zip(self.fom.operator.operators, coefs, DTss)):
            CT = [None for _ in range(self.fom.NCoarse)]
            for j in range(len(self.RBsizeT)):
                for res, coef_ in zip(self.residualT[j].operator.operators, self.residualT[j].operator.coefficients):
                    if is_equal(coef_, coef):
                        CT[j] = res
                        break
            operators.append(TwoScaleBlockOperator(op.A, op.BT, CT, DTs, a=a, b=b, source_spaces=source_spaces,
                                                   range_spaces=range_spaces))
        return LincombOperator(operators, coefs)

    def _construct_residual_rhs(self, a):
        # blocked rhs
        if isinstance(self.fom.rhs, LincombOperator):
            blocks = []
            for block in self.fom.rhs.operators:
                coarse_rhs = block.blocks[0]
                if a == 1:
                    block_rhs = [coarse_rhs]
                else:
                    block_rhs = [coarse_rhs * a]
                blocks.append(SimplifiedBlockColumnOperator(block_rhs, range=self.residual_operator.range))
            blocked_rhs = LincombOperator(blocks, self.fom.rhs.coefficients)
        else:
            coarse_rhs = self.fom.rhs.blocks[0]
            if a == 1:
                block_rhs = [coarse_rhs]
            else:
                block_rhs = [coarse_rhs * a]
            blocked_rhs = SimplifiedBlockColumnOperator(block_rhs, range=self.residual_operator.range)
        return blocked_rhs

    def _construct_residual_product(self):
        return TrueDiagonalBlockOperator(self.product.blocks, True, self.residual_operator.operators[0].range_spaces)

    def _extract_D_matrices(self, coefs):
        DTss = []
        range_spaces = [None for _ in range(len(self.residualT))]
        for coef in coefs:
            DTs = []
            for j, residual in enumerate(self.residualT):
                success = 0
                rhs = residual.rhs
                TPrimes = self.fom.TPrimeCoarsepStartIndices[j] + self.fom.TPrimeCoarsepIndexMap
                coefs_ = residual.operator.coefficients
                for i, coef_ in enumerate(coefs_):
                    if is_equal(coef_, coef):
                        Dij = np.zeros((self.fom.NpCoarse, rhs.range.dim))
                        for l in range(4):
                            TPrime = TPrimes[l]
                            Dij[TPrime] = rhs.operators[len(coefs_) * l + i].matrix.T[0]
                        success = 1
                        break
                if success:
                    op = -NumpyMatrixOperator(sparse.csr_matrix(Dij[self.fom.free])).H
                    DTs.append(op)
                    if range_spaces[j] == None:
                        range_spaces[j] = op.range
                else:
                    DTs.append(None)
            DTss.append(DTs)
        return DTss, range_spaces

    def assemble_error_estimator(self):
        if self.residual_reductor is not None:
            residual = self.residual_reductor.reduce(True)
            error_estimator = CoerciveRBEstimator(residual, tuple(self.residual_reductor.residual_range_dims),
                                            self.coercivity_estimator)
            return error_estimator
        else:
            return None

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

    def __init__(self, blocks, only_first=False, source_spaces=None):
        self.only_first = only_first
        self.blocked_source = True
        self.blocked_range = True
        self.blocks = blocks = np.array(blocks)
        assert 1 <= blocks.ndim <= 2

        # find source/range spaces for every column/row
        self.source_spaces = source_spaces
        self.range_spaces = source_spaces

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
        assert V in self.range
        assert initial_guess is None or initial_guess in self.source and len(initial_guess) == len(V)
        if self.only_first:
            U_blocks = [self.blocks[0].apply_inverse(V.blocks[0], mu=mu,
                                                     initial_guess=(initial_guess.blocks[0]
                                                                    if initial_guess is not None else None),
                                                     least_squares=least_squares)]
            U_blocks.extend(V.blocks[1:])
        else:
            U_blocks = [self.blocks[i].apply_inverse(V.blocks[i], mu=mu,
                                                     initial_guess=(initial_guess.blocks[i]
                                                                    if initial_guess is not None else None),
                                                     least_squares=least_squares)
                        for i in range(self.num_source_blocks)]
        return self.source.make_array(U_blocks)