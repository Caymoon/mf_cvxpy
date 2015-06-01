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

# Methods for SCS iterative solver.

from cvxpy.lin_ops.tree_mat import mul, tmul, sum_dicts
import numpy as np
import scipy.sparse.linalg as LA

def get_mul_funcs(sym_data, constraints):

    def accAmulStoch(x, y, p=None, stoch=False, samples=25):
        if stoch and p == 2:
            y_total = np.zeros(y.shape)
            for k in range(samples):
                y_tmp = np.zeros(y.shape)
                x_tmp = x*np.random.choice([-1, 1], size=x.shape)
                # x_tmp = x*np.random.normal(size=x.shape)
                accAmul(x_tmp, y_tmp, None)
                y_total += np.square(y_tmp)
            y += np.sqrt(y_total/samples)
        else:
            accAmul(x, y, p)

    def accATmulStoch(x, y, p=None, stoch=False, samples=25):
        if stoch and p == 2:
            y_total = np.zeros(y.shape)
            for k in range(samples):
                y_tmp = np.zeros(y.shape)
                x_tmp = x*np.random.choice([-1, 1], size=x.shape)
                # x_tmp = x*np.random.normal(size=x.shape)
                accATmul(x_tmp, y_tmp, None)
                y_total += np.square(y_tmp)
            y += np.sqrt(y_total/samples)
        else:
            accATmul(x, y, p)

    def getDE(D, E):
        pass

    def getM(M):
        pass

    def accAmul(x, y, p=None):
        # y += A*x
        rows = y.shape[0]
        var_dict = vec_to_dict(x, sym_data.var_offsets,
                               sym_data.var_sizes)
        y += constr_mul(constraints, var_dict, rows, p)

    def accATmul(x, y, p=None):
        # y += A.T*x
        terms = constr_unpack(constraints, x)
        val_dict = constr_tmul(constraints, terms, p)
        y += dict_to_vec(val_dict, sym_data.var_offsets,
                         sym_data.var_sizes, sym_data.x_length)

    return (accAmulStoch, accATmulStoch, getDE, getM)

def constr_unpack(constraints, vector):
    """Unpacks a vector into a list of values for constraints.
    """
    values = []
    offset = 0
    for constr in constraints:
        rows, cols = constr.size
        val = np.zeros((rows, cols))
        for col in range(cols):
            val[:, col] = vector[offset:offset+rows]
            offset += rows
        values.append(val)
    return values

def vec_to_dict(vector, var_offsets, var_sizes):
    """Converts a vector to a map of variable id to value.

    Parameters
    ----------
    vector : NumPy matrix
        The vector of values.
    var_offsets : dict
        A map of variable id to offset in the vector.
    var_sizes : dict
        A map of variable id to variable size.

    Returns
    -------
    dict
        A map of variable id to variable value.
    """
    val_dict = {}
    for id_, offset in var_offsets.items():
        size = var_sizes[id_]
        value = np.zeros(size)
        offset = var_offsets[id_]
        for col in range(size[1]):
            value[:, col] = vector[offset:size[0]+offset]
            offset += size[0]
        val_dict[id_] = value
    return val_dict

def dict_to_vec(val_dict, var_offsets, var_sizes, vec_len):
    """Converts a map of variable id to value to a vector.

    Parameters
    ----------
    val_dict : dict
        A map of variable id to value.
    var_offsets : dict
        A map of variable id to offset in the vector.
    var_sizes : dict
        A map of variable id to variable size.
    vector : NumPy matrix
        The vector to store the values in.
    """
    # TODO take in vector.
    vector = np.zeros(vec_len)
    for id_, value in val_dict.items():
        size = var_sizes[id_]
        offset = var_offsets[id_]
        for col in range(size[1]):
            # Handle scalars separately.
            if np.isscalar(value):
                vector[offset:size[0]+offset] = value
            else:
                vector[offset:size[0]+offset] = np.squeeze(value[:, col])
            offset += size[0]
    return vector

def constr_mul(constraints, var_dict, vec_size, p=None):
    """Multiplies a vector by the matrix implied by the constraints.

    Parameters
    ----------
    constraints : list
        A list of linear constraints.
    var_dict : dict
        A dictionary mapping variable id to value.
    vec_size : int
        The length of the product vector.
    p : int
        The p-norm being approximated.
    """
    product = np.zeros(vec_size)
    offset = 0
    for constr in constraints:
        result = mul(constr.expr, var_dict, p)
        rows, cols = constr.size
        # Handle scalars separately.
        if np.isscalar(result):
            product[offset:offset+rows*cols] = result
        else:
            flattened = np.reshape(result, rows*cols, order='F')
            product[offset:offset+rows*cols] = flattened
        offset += rows*cols
    return product

def constr_tmul(constraints, values, p=None):
    """Multiplies a vector by the transpose of the constraints matrix.

    Parameters
    ----------
    constraints : list
        A list of linear constraints.
    values : list
        A list of NumPy matrices.
    p : int
        The p-norm being approximated.

    Returns
    -------
    dict
        A mapping of variable id to value.
    """
    products = []
    for constr, val in zip(constraints, values):
        products.append(tmul(constr.expr, val, p))
    return sum_dicts(products, p)
