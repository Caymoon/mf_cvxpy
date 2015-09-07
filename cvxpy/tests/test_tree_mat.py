"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from cvxpy import *
import cvxpy.settings as s
from cvxpy.lin_ops.tree_mat import mul, tmul, prune_constants
import cvxpy.problems.iterative as iterative
from cvxpy.problems.solvers.utilities import SOLVERS
from cvxpy.problems.problem_data.sym_data import SymData
import numpy as np
import numpy.linalg
import scipy.sparse as sp
import scipy.linalg as LA
import unittest
from base_test import BaseTest


def pnorm_mat_mul(A, B, p):
    """p-norm multiplication of two matrices.
    """
    rows = A.shape[0]
    cols = B.shape[1]
    result = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            total = 0
            for k in range(A.shape[1]):
                total += np.power(np.abs(A[i,k]*B[k,j]), p)
            result[i,j] = total**(1.0/p)
    return result

class test_tree_mat(BaseTest):
    """ Unit tests for the matrix ops with expression trees. """

    def test_mul(self):
        """Test the mul method.
        """
        n = 2
        ones = np.mat(np.ones((n, n)))
        # Multiplication
        x = Variable(n, n)
        A = np.matrix("1 2; 3 4")
        expr = (A*x).canonical_form[0]

        val_dict = {x.id: ones}

        result = mul(expr, val_dict)
        assert (result == A*ones).all()

        result_dict = tmul(expr, result)
        assert (result_dict[x.id] == A.T*A*ones).all()

        # Multiplication with promotion.
        t = Variable()
        A = np.matrix("1 2; 3 4")
        expr = (A*t).canonical_form[0]

        val_dict = {t.id: 2}

        result = mul(expr, val_dict)
        assert (result == A*2).all()

        result_dict = tmul(expr, result)
        total = 0
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                total += A[i, j]*result[i, j]
        assert (result_dict[t.id] == total)

        # Addition
        y = Variable(n, n)
        expr = (y + A*x).canonical_form[0]
        val_dict = {x.id: np.ones((n, n)),
                    y.id: np.ones((n, n))}

        result = mul(expr, val_dict)
        assert (result == A*ones + ones).all()

        result_dict = tmul(expr, result)
        assert (result_dict[y.id] == result).all()
        assert (result_dict[x.id] == A.T*result).all()

        val_dict = {x.id: A,
                    y.id: A}

        # Indexing
        expr = (x[:, 0] + y[:, 1]).canonical_form[0]
        result = mul(expr, val_dict)
        assert (result == A[:, 0] + A[:, 1]).all()

        result_dict = tmul(expr, result)
        mat = ones
        mat[:, 0] = result
        mat[:, 1] = 0
        assert (result_dict[x.id] == mat).all()

        # Negation
        val_dict = {x.id: A}
        expr = (-x).canonical_form[0]

        result = mul(expr, val_dict)
        assert (result == -A).all()

        result_dict = tmul(expr, result)
        assert (result_dict[x.id] == A).all()

        # Transpose
        expr = x.T.canonical_form[0]
        val_dict = {x.id: A}
        result = mul(expr, val_dict)
        assert (result == A.T).all()
        result_dict = tmul(expr, result)
        assert (result_dict[x.id] == A).all()

        # Convolution
        x = Variable(3)
        f = np.matrix(np.array([1, 2, 3])).T
        g = np.array([0, 1, 0.5])
        f_conv_g = np.array([ 0., 1., 2.5,  4., 1.5])
        expr = conv(f, x).canonical_form[0]
        val_dict = {x.id: g}
        result = mul(expr, val_dict)
        self.assertItemsAlmostEqual(result, f_conv_g)
        value = np.array(range(5))
        result_dict = tmul(expr, value)
        toep = LA.toeplitz(np.array([1,0,0]),
                           np.array([1, 2, 3, 0, 0]))
        x_val = toep.dot(value)
        self.assertItemsAlmostEqual(result_dict[x.id], x_val)

        # Elementwise multiplication.
        x = Variable(n, n)
        A = np.matrix("1 -2; -3 4").A
        expr = (mul_elemwise(A, x)).canonical_form[0]

        val_dict = {x.id: np.ones((n, n))}

        result = mul(expr, val_dict)
        assert (result == A).all()

        result_dict = tmul(expr, result)
        assert (result_dict[x.id] == A*A).all()

        # Reshape.
        x = Variable(3, 2)
        expr = (reshape(x, 1, 6)).canonical_form[0]
        orig = np.array([[1,2,3], [4, 5, 6]])
        val_dict = {x.id: orig}

        result = mul(expr, val_dict)
        assert (result == np.matrix([[1, 4, 2, 5, 3, 6]])).all()

        result_dict = tmul(expr, result)
        assert (np.matrix(result_dict[x.id]) == np.matrix([[1,4,2], [5, 3, 6]]).T).all()

        # Vstack.
        x = Variable(2, 3)
        y = Variable(1, 3)
        expr = (vstack(x, y)).canonical_form[0]
        orig_x = np.array([[1,2,3], [4, 5, 6]])
        orig_y = np.array([[7, 8, 9]])
        val_dict = {x.id: orig_x, y.id: orig_y}

        result = mul(expr, val_dict)
        assert (result == np.array([[1,2,3], [4, 5, 6], [7, 8, 9]])).all()

        result_dict = tmul(expr, result)
        print np.matrix(result_dict[x.id])
        assert (np.matrix(result_dict[x.id]) == np.array([[1,2,3], [4, 5, 6]])).all()
        assert (np.matrix(result_dict[y.id]) == np.array([[7, 8, 9]])).all()


    def test_pmul(self):
        """Test the p mul method.
        """
        n = 2
        ones = np.mat(np.ones((n, n)))
        # Multiplication
        x = Variable(n, n)
        A = np.matrix("-1 2; -3 4")
        expr = (A*x).canonical_form[0]

        val_dict = {x.id: ones}

        for p in [1, 2, 3]:
            result = mul(expr, val_dict, p)
            self.assertItemsEqual(np.array(result)[:,0], numpy.linalg.norm(A, p, axis=1))

            result_dict = tmul(expr, ones, p)
            assert (np.array(result_dict[x.id])[:,0] == numpy.linalg.norm(A, p, axis=0)).all()

        # Multiplication with promotion.
        t = Variable()
        A = np.matrix("1 -2; -3 -4")
        expr = (A*t).canonical_form[0]
        ones = np.mat(np.ones((n, n)))

        val_dict = {t.id: 2}

        for p in [1, 2, 3]:
            result = mul(expr, val_dict, p)
            assert np.allclose(result, np.abs(A)*2)

            result_dict = tmul(expr, result, p)
            output = pnorm_mat_mul(A.T, result, p)
            final = np.linalg.norm(np.diag(output), p)
            assert np.allclose(result_dict[t.id], final)

        # Addition
        y = Variable(n, n)
        expr = (y + x).canonical_form[0]
        val_dict = {x.id: 2*np.ones((n, n)),
                    y.id: np.ones((n, n))}

        for p in [1, 2, 3]:
            result = mul(expr, val_dict, p)
            assert np.allclose(result,
                               np.power((2**p + 1)*ones, 1.0/p))

            result_dict = tmul(expr, result, p)
            assert (result_dict[y.id] == result).all()
            assert (result_dict[x.id] == result).all()

        abs_A = np.abs(A)
        val_dict = {x.id: abs_A,
                    y.id: abs_A}

        # Indexing
        expr = (x[:, 0] + y[:, 1]).canonical_form[0]
        for p in [1, 2, 3]:
            result = mul(expr, val_dict, p)
            assert np.allclose(result,
                       np.power(np.power(abs_A[:, 0], p) +
                                np.power(abs_A[:, 1], p),
                                1.0/p)
                   )

            result_dict = tmul(expr, result, p)
            mat = ones
            mat[:, 0] = result
            mat[:, 1] = 0
            assert (result_dict[x.id] == mat).all()

        # Negation
        val_dict = {x.id: abs_A}
        expr = (-x).canonical_form[0]
        for p in [1, 2, 3]:
            result = mul(expr, val_dict, p)
            assert np.allclose(result, abs_A)

            result_dict = tmul(expr, result, p)
            assert np.allclose(result_dict[x.id], abs_A)

        # Transpose
        expr = x.T.canonical_form[0]
        for p in [1, 2, 3]:
            val_dict = {x.id: abs_A}
            result = mul(expr, val_dict, p)
            assert np.allclose(result, abs_A.T)
            result_dict = tmul(expr, result, p)
            assert np.allclose(result_dict[x.id], abs_A)

        # Convolution
        x = Variable(3)
        f = np.matrix(np.array([1, 2, 3])).T
        g = np.array([0, 1, 0.5])
        expr = conv(f, x).canonical_form[0]
        val_dict = {x.id: g}
        for p in [1, 2, 3]:
            result = mul(expr, val_dict, p)
            f_conv_g = np.power(conv(np.power(f, p), np.power(g, p)).value, 1.0/p)
            self.assertItemsAlmostEqual(result, f_conv_g)
            value = np.array(range(5))
            result_dict = tmul(expr, value, p)
            toep = LA.toeplitz(np.array([1,0,0]),
                               np.array([1, 2, 3, 0, 0]))
            x_val = pnorm_mat_mul(toep, np.matrix(value).T, p)
            self.assertItemsAlmostEqual(result_dict[x.id], x_val)

        # Elementwise multiplication.
        n = 2
        x = Variable(n, n)
        A = np.matrix("1 -2; -3 4").A
        expr = (mul_elemwise(A, x)).canonical_form[0]
        val_dict = {x.id: np.ones((n, n))}
        for p in [1, 2, 3]:
            result = mul(expr, val_dict, p)
            assert (result == np.abs(A)).all()

            result_dict = tmul(expr, result, p)
            self.assertItemsAlmostEqual(result_dict[x.id], A*A)

        for p in [1, 2, 3]:
            # Reshape.
            x = Variable(3, 2)
            expr = (reshape(x, 1, 6)).canonical_form[0]
            orig = np.array([[1,2,3], [4, 5, 6]])
            val_dict = {x.id: orig}

            result = mul(expr, val_dict, p)
            assert (result == np.matrix([[1, 4, 2, 5, 3, 6]])).all()

            result_dict = tmul(expr, result, p)
            assert np.allclose(np.matrix(result_dict[x.id]), np.matrix([[1.,4.,2.], [5., 3., 6.]]).T)

        for p in [1, 2, 3]:
            # Vstack.
            x = Variable(2, 3)
            y = Variable(1, 3)
            expr = (vstack(x, y)).canonical_form[0]
            orig_x = np.array([[1,2,3], [4, 5, 6]])
            orig_y = np.array([[7, 8, 9]])
            val_dict = {x.id: orig_x, y.id: orig_y}

            result = mul(expr, val_dict, p)
            assert (result == np.array([[1,2,3], [4, 5, 6], [7, 8, 9]])).all()

            result_dict = tmul(expr, result, p)
            assert np.allclose(np.matrix(result_dict[x.id]), np.array([[1.,2.,3.], [4., 5., 6.]]))
            assert np.allclose(np.matrix(result_dict[y.id]), np.array([[7., 8., 9.]]))

    def test_prune_constants(self):
        """Test pruning constants from constraints.
        """
        x = Variable(2)
        A = np.matrix("1 2; 3 4")
        constraints = (A*x <= 2).canonical_form[1]
        pruned = prune_constants(constraints)
        prod = mul(pruned[0].expr, {})
        self.assertItemsAlmostEqual(prod, np.zeros(A.shape[0]))

        # Test no-op
        constraints = (0*x <= 2).canonical_form[1]
        pruned = prune_constants(constraints)
        prod = mul(pruned[0].expr, {x.id: 1})
        self.assertItemsAlmostEqual(prod, np.zeros(A.shape[0]))

    def test_mul_funcs(self):
        """Test functions to multiply by A, A.T
        """
        n = 10
        x = Variable(n)
        obj = Minimize(norm(x, 1))
        constraints = [x >= 2]
        prob = Problem(obj, constraints)
        data, dims = prob.get_problem_data(solver=SCS)
        A = data["A"]
        objective, constraints = prob.canonicalize()
        sym_data = SymData(objective, constraints, SOLVERS[SCS])
        constraints = prune_constants(sym_data.constraints)
        Amul, ATmul, _, _ = iterative.get_mul_funcs(sym_data, constraints)
        vec = np.array(range(sym_data.x_length))
        # A*vec
        result = np.zeros(A.shape[0])
        Amul(vec, result)
        self.assertItemsAlmostEqual(A*vec, result)
        Amul(vec, result)
        self.assertItemsAlmostEqual(2*A*vec, result)
        # A.T*vec
        vec = np.array(range(A.shape[0]))
        result = np.zeros(A.shape[1])
        ATmul(vec, result)
        self.assertItemsAlmostEqual(A.T*vec, result)
        ATmul(vec, result)
        self.assertItemsAlmostEqual(2*A.T*vec, result)
        for p in [1, 2, 3]:
            # |A|*vec
            vec = np.array(range(A.shape[1]))
            result = np.zeros(A.shape[0])
            Amul(vec, result, p)
            self.assertItemsAlmostEqual(pnorm_mat_mul(A, np.matrix(vec).T, p), result)
            # |A.T|*vec
            vec = np.array(range(A.shape[0]))
            result = np.zeros(A.shape[1])
            ATmul(vec, result, p)
            other_result = pnorm_mat_mul(A.T, np.matrix(vec).T, p)
            self.assertItemsAlmostEqual(other_result, result)
