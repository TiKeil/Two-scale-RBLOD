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

"""
Optimized rom solve
"""
import numpy as np
import dill
from pymor.core.base import ImmutableObject
from pymor.core.exceptions import InversionError
from pymor.operators.constructions import LincombOperator


class OptimizedNumpyModelStage1(ImmutableObject):
    def __init__(self, rom, Kij=None, T=None):
        self.rom = rom
        self.Kij = Kij
        self.T = T
        self.solution_space = self.rom.solution_space
        self.parameters = rom.parameters

        DoFs = self.rom.operator.operators[0].matrix.shape[0]
        Operator_array = np.zeros((len(self.rom.operator.operators), DoFs, DoFs))
        for (i, op) in enumerate(self.rom.operator.operators):
            Operator_array[i] = op.matrix
        self.operator_array = Operator_array

        if isinstance(self.rom.rhs, LincombOperator):
            rhs_array = np.zeros((len(self.rom.rhs.operators), DoFs, 1))
            for (i, op) in enumerate(self.rom.rhs.operators):
                rhs_array[i] = op.matrix
            self.rhs_array = rhs_array
        else:
            rhs_array = np.zeros((1, DoFs, 1))
            rhs_array[0] = self.rom.rhs.matrix
            self.rhs_array = rhs_array

        if isinstance(self.rom.output_functional.operators[0], LincombOperator):
            pass
        else:
            output_array = np.zeros(
                (
                    len(self.rom.output_functional.operators),
                    self.rom.output_functional.operators[0].matrix.shape[0],
                    self.rom.output_functional.operators[0].matrix.shape[1],
                )
            )
            for (i, op) in enumerate(self.rom.output_functional.operators):
                output_array[i] = op.matrix
            self.output_array = output_array

        if Kij is not None:
            Kij_array = np.zeros((len(self.Kij.operators), self.Kij.operators[0].value.dim))
            for (i, op) in enumerate(self.Kij.operators):
                Kij_array[i] = op.value.to_numpy()[0]
        self.Kij_array = Kij_array if Kij is not None else None

        if isinstance(self.rom.rhs, LincombOperator):
            self.rhs_coefficients = self.rom.rhs.coefficients
        else:
            self.rhs_coefficients = [lambda mu: 1.0]
        self.op_coefficients = self.rom.operator.coefficients
        self.output_coefficients = self.rom.output_functional.coefficients
        self.Kij_coefficients = Kij.coefficients if Kij is not None else None

    def solve(self, mu):
        lhs, rhs = self._assemble(mu)
        return self.solution_space.from_numpy(np.linalg.solve(lhs, rhs))

    def output(self, mu=None, return_solution=False):
        solution = self.solve(mu)
        if self.rom is not None:
            if isinstance(self.rom.output_functional.operators[0], LincombOperator):
                # can be speeded up further but is not used in publication
                output = self.rom.output_functional.apply(solution, mu).to_numpy()
            else:
                evaluated_mus_output = [c.evaluate(mu) if hasattr(c, "evaluate") else c
                                        for c in self.output_coefficients]
                output = np.einsum("tij,t,j->i", self.output_array, evaluated_mus_output, solution.to_numpy()[0])
                output = output.reshape(1, len(output))
        else:
            evaluated_mus_output = [c.evaluate(mu) if hasattr(c, "evaluate") else c for c in self.output_coefficients]
            output = np.einsum("tij,t,j->i", self.output_array, evaluated_mus_output, solution.to_numpy()[0])
            output = output.reshape(1, len(output))
        if return_solution:
            return output, solution
        else:
            return output

    def _assemble(self, mu):
        evaluated_mus_lhs = [c.evaluate(mu) if hasattr(c, "evaluate") else c for c in self.op_coefficients]
        evaluated_mus_rhs = [c.evaluate(mu) if hasattr(c, "evaluate") else c for c in self.rhs_coefficients]
        lhs = np.einsum("tij,t->ij", self.operator_array, evaluated_mus_lhs)
        rhs = np.einsum("tij,t->ij", self.rhs_array, evaluated_mus_rhs).flatten()
        return lhs, rhs

    def Kij_constant(self, mu):
        evaluated_mus_kij = [c.evaluate(mu) if hasattr(c, "evaluate") else c for c in self.Kij_coefficients]
        Kij = np.einsum("ti,t->i", self.Kij_array, evaluated_mus_kij)
        return Kij

    def minimal_object(self, add_error_residual=True):
        if isinstance(self.rom.output_functional.operators[0], LincombOperator):
            assert 0, "you can not use the minimal optimized model for this case"
        error_residual = self.rom.error_estimator.residual if add_error_residual else None
        returning_residual = self.rom.error_estimator.residual if not add_error_residual else None
        if hasattr(self.rom.error_estimator, 'coercivity_estimator'):
            coercivity_estimator = self.rom.error_estimator.coercivity_estimator
        else:
            coercivity_estimator = None
        return MinimalOptimizedNumpyModelStage1(self.operator_array, self.rhs_array, self.output_array, error_residual,
                                                coercivity_estimator, self.solution_space,
                                                self.op_coefficients, self.rhs_coefficients, self.output_coefficients,
                                                self.Kij_coefficients, self.Kij_array, self.parameters,
                                                T=self.T),\
               returning_residual

    def estimate_error(self, mu, store_in_tmp=False):
        U = self.solve(mu)
        if self.rom is None:
            if self.error_residual is None:
                assert store_in_tmp
                dbfile = open(f'{store_in_tmp}/err_res_{self.T}', "rb")
                error_residual = dill.load(dbfile)
            else:
                error_residual = self.error_residual
            est = error_residual.apply(U, mu).norm()/self.coercivity_estimator(mu)
        else:
            est = self.rom.estimate_error(U, mu)
        return est

class MinimalOptimizedNumpyModelStage1(OptimizedNumpyModelStage1):
    def __init__(self, operator_array, rhs_array, output_array, error_residual, coercivity_estimator,
                 solution_space, op_coefficients, rhs_coefficients, output_coefficients,
                 Kij_coefficients, Kij_array, parameters, T):
        self.rom = None
        self.__auto_init(locals())

class OptimizedTwoScaleNumpyModel(ImmutableObject):
    def __init__(self, rom):
        self.rom = rom
        self.parameters = rom.parameters

        DoFs_0 = self.rom.operator.operators[0].matrix.shape[0]
        DoFs_1 = self.rom.operator.operators[0].matrix.shape[1]
        operator_array = np.zeros((len(self.rom.operator.operators), DoFs_0, DoFs_1))
        for (i, op) in enumerate(self.rom.operator.operators):
            operator_array[i] = op.matrix
        self.operator_array = operator_array

        if isinstance(self.rom.rhs, LincombOperator):
            rhs_array = np.zeros((len(self.rom.rhs.operators), DoFs_0, 1))
            for (i, op) in enumerate(self.rom.rhs.operators):
                rhs_array[i] = op.matrix
            self.rhs_array = rhs_array
        else:
            rhs_array = np.zeros((1, DoFs_0, 1))
            rhs_array[0] = self.rom.rhs.matrix
            self.rhs_array = rhs_array

    def solve(self, mu, least_squares=False):
        lhs, rhs = self._assemble(mu)
        if least_squares:
            try:
                R, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
            except np.linalg.LinAlgError as e:
                raise InversionError(f"{str(type(e))}: {str(e)}")
            R = R.T
        else:
            R = np.linalg.solve(lhs, rhs)
        return self.rom.solution_space.from_numpy(R.flatten())

    def output(self, mu):
        return NotImplemented

    def _assemble(self, mu):
        evaluated_mus_lhs = [c.evaluate(mu) if hasattr(c, "evaluate") else c for c in self.rom.operator.coefficients]
        if isinstance(self.rom.rhs, LincombOperator):
            evaluated_mus_rhs = [c.evaluate(mu) if hasattr(c, "evaluate") else c for c in self.rom.rhs.coefficients]
        else:
            evaluated_mus_rhs = [1.0]
        lhs = np.einsum("tij,t->ij", self.operator_array, evaluated_mus_lhs)
        rhs = np.einsum("tij,t->ij", self.rhs_array, evaluated_mus_rhs)
        return lhs, rhs
