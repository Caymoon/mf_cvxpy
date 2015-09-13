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

import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import copy
import numpy as np
import numpy.linalg
import scipy.sparse as sp
from numpy.fft import fft, ifft
from scipy.signal import fftconvolve

# Utility functions for treating an expression tree as a matrix
# and multiplying by it and it's transpose.

def mul(lin_op, val_dict, p=None):
    """Multiply the expression tree by a vector.

    Parameters
    ----------
    lin_op : LinOp
        The root of an expression tree.
    val_dict : dict
        A map of variable id to value.
    p : int, optional
        The p-norm being approximated.

    Returns
    -------
    NumPy matrix
        The result of the multiplication.
    """
    # Look up the value for a variable.
    if lin_op.type == lo.VARIABLE:
        if lin_op.data in val_dict:
            return val_dict[lin_op.data]
        # Defaults to zero if no value given.
        else:
            return np.mat(np.zeros(lin_op.size))
    # Return all zeros for NO_OP.
    elif lin_op.type == lo.NO_OP:
        return np.mat(np.zeros(lin_op.size))
    else:
        eval_args = []
        for arg in lin_op.args:
            eval_args.append(mul(arg, val_dict, p))
        if p is not None:
            return op_pmul(lin_op, eval_args, p)
        else:
            return op_mul(lin_op, eval_args)

def tmul(lin_op, value, p=None):
    """Multiply the transpose of the expression tree by a vector.

    Parameters
    ----------
    lin_op : LinOp
        The root of an expression tree.
    value : NumPy matrix
        The vector to multiply by.
    p : int, optional
        The p-norm being approximated.

    Returns
    -------
    dict
        A map of variable id to value.
    """
    # Store the value as the variable.
    if lin_op.type == lo.VARIABLE:
        return {lin_op.data: value}
    # Do nothing for NO_OP.
    elif lin_op.type == lo.NO_OP:
        return {}
    else:
        if p is not None:
            results = op_ptmul(lin_op, value, p)
        else:
            results = op_tmul(lin_op, value)
        result_dicts = []
        for arg, result in zip(lin_op.args, results):
            result_dicts.append(tmul(arg, result, p))
        # Sum repeated ids.
        return sum_dicts(result_dicts, p)

def sum_dicts(dicts, p):
    """Sums the dictionaries entrywise.

    Parameters
    ----------
    dicts : list
        A list of dictionaries with numeric entries.
    p : int
        The p-norm being approximated.

    Returns
    -------
    dict
        A dict with the sum.
    """
    # Sum repeated entries.
    sum_dict = {}
    for val_dict in dicts:
        for id_, value in val_dict.items():
            if p is None:
                sum_dict[id_] = value + sum_dict.get(id_, 0)
            else:
                value = np.power(np.abs(value), p)
                sum_dict[id_] = value + sum_dict.get(id_, 0)
    # Entrywise ^(1/p)
    if p is not None:
        for id_, value in sum_dict.items():
            sum_dict[id_] = np.power(value, 1.0/p)

    return sum_dict

def op_mul(lin_op, args):
    """Applies the linear operator to the arguments.

    Parameters
    ----------
    lin_op : LinOp
        A linear operator.
    args : list
        The arguments to the operator.

    Returns
    -------
    NumPy matrix or SciPy sparse matrix.
        The result of applying the linear operator.
    """
    # Constants convert directly to their value.
    if lin_op.type in [lo.SCALAR_CONST, lo.DENSE_CONST, lo.SPARSE_CONST]:
        result = lin_op.data
    elif lin_op.type == lo.PARAM:
        result = lin_op.data.value
    # No-op is not evaluated.
    elif lin_op.type == lo.NO_OP:
        return None
    # For non-leaves, recurse on args.
    elif lin_op.type == lo.SUM:
        result = sum(args)
    elif lin_op.type == lo.NEG:
        result = -args[0]
    elif lin_op.type == lo.MUL:
        coeff = mul(lin_op.data, {})
        result = coeff*args[0]
    elif lin_op.type == lo.RMUL:
        coeff = mul(lin_op.data, {})
        result = args[0]*coeff
    elif lin_op.type == lo.MUL_ELEM:
        coeff = mul(lin_op.data, {})
        result = np.multiply(args[0], coeff)
    elif lin_op.type == lo.DIV:
        divisor = mul(lin_op.data, {})
        result = args[0]/divisor
    elif lin_op.type == lo.SUM_ENTRIES:
        result = np.sum(args[0])
    elif lin_op.type == lo.INDEX:
        row_slc, col_slc = lin_op.data
        result = args[0][row_slc, col_slc]
    elif lin_op.type == lo.TRANSPOSE:
        result = args[0].T
    elif lin_op.type in [lo.CONV, lo.CONV2D]:
        result = conv_mul(lin_op, args[0])
    elif lin_op.type == lo.PROMOTE:
        result = np.ones(lin_op.size)*args[0]
    elif lin_op.type == lo.DIAG_VEC:
        val = intf.from_2D_to_1D(args[0])
        result = np.diag(val)
    elif lin_op.type == lo.RESHAPE:
        result = np.reshape(args[0], lin_op.size, order='F')
    elif lin_op.type == lo.VSTACK:
        result = np.vstack(args)
    else:
        raise Exception("Unknown linear operator %s." % lin_op.type)
    return result

def elem_power(mat, p):
    """Elementwise power.
    """
    if np.isscalar(mat):
        mat = np.abs(mat)**p
    elif sp.issparse(mat):
        mat = mat.copy()
        np.abs(mat.data, mat.data)
        np.power(mat.data, p, mat.data)
    else:
        mat = mat.copy()
        np.abs(mat, mat)
        np.power(mat, p, mat)
    return mat

def mat_pmul(lh_mat, rh_mat, p):
    """Matrix multiplication with p-norms.
    """
    lh_mat = elem_power(lh_mat, p)
    rh_mat = elem_power(rh_mat, p)
    # TODO is * always right?
    result = lh_mat*rh_mat
    return elem_power(result, 1.0/p)

def op_pmul(lin_op, args, p):
    """Applies the p-norm approximation of the linear operator to the arguments.

    Parameters
    ----------
    lin_op : LinOp
        A linear operator.
    args : list
        The arguments to the operator.
    p : int
        The p-norm being approximated.

    Returns
    -------
    NumPy matrix or SciPy sparse matrix.
        The result of applying the linear operator.
    """
    # Take the elementwise p-norm.
    if lin_op.type == lo.SUM:
        result = 0
        for arg in args:
            result += np.power(arg, p)
        result = np.power(result, 1.0/p)
    elif lin_op.type == lo.MUL_ELEM:
        coeff = mul(lin_op.data, {})
        coeff = elem_power(coeff, 1)
        result = np.multiply(coeff, args[0])
    # Take the p-norm.
    elif lin_op.type == lo.SUM_ENTRIES:
        result = numpy.linalg.norm(args[0], p)
    elif lin_op.type == lo.NEG:
        result = args[0]
    # (|A|^p x^p)^(1/p)
    elif lin_op.type == lo.MUL:
        coeff = mul(lin_op.data, {})
        result = mat_pmul(coeff, args[0], p)
    # (x^p|A|^p)^(1/p)
    elif lin_op.type == lo.RMUL:
        coeff = mul(lin_op.data, {})
        result = mat_pmul(args[0], coeff, p)
    elif lin_op.type == lo.MUL_ELEM:
        coeff = mul(lin_op.data, {})
        result = mat_pmul(args[0], coeff, p)
    # x/|a|
    elif lin_op.type == lo.DIV:
        divisor = np.abs(mul(lin_op.data, {}))
        result = args[0]/divisor
    elif lin_op.type == lo.CONV:
        result = conv_mul(lin_op, args[0], False, p)
    else:
        result = op_mul(lin_op, args)
    return result

def op_tmul(lin_op, value):
    """Applies the transpose of the linear operator to the arguments.

    Parameters
    ----------
    lin_op : LinOp
        A linear operator.
    value : NumPy matrix
        A numeric value to apply the operator's transpose to.

    Returns
    -------
    NumPy matrix or SciPy sparse matrix.
        The result of applying the linear operator.
    """
    if lin_op.type == lo.SUM:
        result = len(lin_op.args)*[value]
    elif lin_op.type == lo.VSTACK:
        result = []
        offset = 0
        for arg in lin_op.args:
            rows, cols = arg.size
            result.append(value[offset:offset+rows, 0:cols])
            offset += rows
    elif lin_op.type == lo.NEG:
        result = -value
    elif lin_op.type == lo.MUL_ELEM:
        coeff = mul(lin_op.data, {})
        result = np.multiply(coeff, value)
    elif lin_op.type == lo.MUL:
        coeff = mul(lin_op.data, {})
        # Scalar coefficient, no need to transpose.
        if np.isscalar(coeff):
            result = coeff*value
        else:
            result = coeff.T*value
    elif lin_op.type == lo.RMUL:
        coeff = mul(lin_op.data, {})
        # Scalar coefficient, no need to transpose.
        if np.isscalar(coeff):
            result = value*coeff
        else:
            result = value*coeff.T
    elif lin_op.type == lo.DIV:
        divisor = mul(lin_op.data, {})
        result = value/divisor
    elif lin_op.type == lo.SUM_ENTRIES:
        result = np.mat(np.ones(lin_op.args[0].size))*value
    elif lin_op.type == lo.INDEX:
        row_slc, col_slc = lin_op.data
        result = np.mat(np.zeros(lin_op.args[0].size))
        result[row_slc, col_slc] = value
    elif lin_op.type == lo.TRANSPOSE:
        result = value.T
    elif lin_op.type == lo.PROMOTE:
        result = np.sum(value)
    elif lin_op.type == lo.DIAG_VEC:
        result = np.diag(value)
    elif lin_op.type == lo.CONV:
        result = conv_mul(lin_op, value, transpose=True)
    elif lin_op.type == lo.RESHAPE:
        result = np.reshape(value, lin_op.args[0].size, order='F')
    else:
        raise Exception("Unknown linear operator %s." % lin_op.type)

    if isinstance(result, list):
        return result
    else:
        return [result]

def op_ptmul(lin_op, value, p):
    """Applies the linear operator ~||A.T||_p to the arguments.

    Parameters
    ----------
    lin_op : LinOp
        A linear operator.
    value : NumPy matrix
        A numeric value to apply the operator's transpose to.
    p : int
        The p-norm being approximated.

    Returns
    -------
    NumPy matrix or SciPy sparse matrix.
        The result of applying the linear operator.
    """
    if lin_op.type == lo.NEG:
        result = value
    elif lin_op.type == lo.MUL_ELEM:
        coeff = mul(lin_op.data, {})
        coeff = elem_power(coeff, 1)
        result = np.multiply(coeff, value)
    # Take the p-norm.
    elif lin_op.type == lo.PROMOTE:
        result = numpy.linalg.norm(value, p)
    # (|A^T|^p x^p)^(1/p)
    elif lin_op.type == lo.MUL:
        coeff = mul(lin_op.data, {})
        # If scalar coefficient, no need to transpose.
        if not np.isscalar(coeff):
            coeff = coeff.T
        result = mat_pmul(coeff, value, p)
    # (x^p|A^T|^p)^(1/p)
    elif lin_op.type == lo.RMUL:
        coeff = mul(lin_op.data, {})
        # If scalar coefficient, no need to transpose.
        if not np.isscalar(coeff):
            coeff = coeff.T
        result = mat_pmul(value, coeff, p)
    # x/|a|
    elif lin_op.type == lo.DIV:
        divisor = np.abs(mul(lin_op.data, {}))
        result = value/divisor
    elif lin_op.type == lo.CONV:
        result = conv_mul(lin_op, value, True, p)
    else:
        result = op_tmul(lin_op, value)

    if isinstance(result, list):
        return result
    else:
        return [result]

def conv_mul(lin_op, rh_val, transpose=False, p=None):
    """Multiply by a convolution operator.

    arameters
    ----------
    lin_op : LinOp
        The root linear operator.
    rh_val : NDArray
        The vector being convolved.
    transpose : bool
        Is the transpose of convolution being applied?
    p : int, optional
        The p-norm being used.

    Returns
    -------
    NumPy NDArray
        The convolution.
    """
    constant = mul(lin_op.data, {})
    # Convert to 2D
    constant, rh_val = map(intf.from_1D_to_2D, [constant, rh_val])
    # c => |c|^p, x => x^p.
    if p is not None:
        constant = np.power(np.abs(constant), p)
        rh_val = np.power(np.abs(rh_val), p)

    if transpose:
        constant = np.flipud(constant)
        # rh_val always larger than constant.
        result = fftconvolve(rh_val, constant, mode='valid')
    else:
        # First argument must be larger.
        if constant.size >= rh_val.size:
            result = fftconvolve(constant, rh_val, mode='full')
        else:
            result = fftconvolve(rh_val, constant, mode='full')

    # result => result^(1/p)
    if p is not None:
        result = np.power(result, 1.0/p)

    return result

def get_constant(lin_op):
    """Returns the constant term in the expression.

    Parameters
    ----------
    lin_op : LinOp
        The root linear operator.

    Returns
    -------
    NumPy NDArray
        The constant term as a flattened vector.
    """
    constant = mul(lin_op, {})
    const_size = constant.shape[0]*constant.shape[1]
    return np.reshape(constant, const_size, 'F')

def get_constr_constant(constraints):
    """Returns the constant term for the constraints matrix.

    Parameters
    ----------
    constraints : list
        The constraints that form the matrix.

    Returns
    -------
    NumPy NDArray
        The constant term as a flattened vector.
    """
    # TODO what if constraints is empty?
    constants = [get_constant(c.expr) for c in constraints]
    return np.hstack(constants)

def prune_constants(constraints):
    """Returns a new list of constraints with constant terms removed.

    Parameters
    ----------
    constraints : list
        The constraints that form the matrix.

    Returns
    -------
    list
        The pruned constraints.
    """
    pruned_constraints = []
    for constr in constraints:
        constr_type = type(constr)
        expr = copy.deepcopy(constr.expr)
        is_constant = prune_expr(expr)
        # Replace a constant root with a NO_OP.
        if is_constant:
            expr = lo.LinOp(lo.NO_OP, expr.size, [], None)
        pruned = constr_type(expr, constr.constr_id, constr.size)
        pruned_constraints.append(pruned)
    return pruned_constraints

def prune_expr(lin_op):
    """Prunes constant branches from the expression.

    Parameters
    ----------
    lin_op : LinOp
        The root linear operator.

    Returns
    -------
    bool
        Were all the expression's arguments pruned?
    """
    if lin_op.type == lo.VARIABLE:
        return False
    elif lin_op.type in [lo.SCALAR_CONST,
                         lo.DENSE_CONST,
                         lo.SPARSE_CONST,
                         lo.PARAM]:
        return True

    pruned_args = []
    is_constant = True
    for arg in lin_op.args:
        arg_constant = prune_expr(arg)
        if arg_constant:
            arg = lo.LinOp(lo.NO_OP, arg.size, [], None)
        else:
            is_constant = False
        # Special case for sum.
        # TODO do this in simplify_dag.
        if not (arg_constant and lin_op.type == lo.SUM):
            pruned_args.append(arg)
    # Overwrite old args with only non-constant args.
    lin_op.args[:] = pruned_args[:]
    return is_constant

def combine_lin_ops(operators):
    """Combines the LinOps by stacking their output into a vector.

    Parameters
    ----------
    operators : list
        A list of LinOps.

    Returns
    -------
    LinOp
        The combined LinOp.
    """
    # First vectorize all the LinOp outputs.
    vect_lin_ops = []
    total_length = 0
    for lin_op in operators:
        if lin_op.size[1] != 1:
            new_size = (lin_op.size[0]*lin_op.size[1], 1)
            lin_op = lu.reshape(lin_op, new_size)
        vect_lin_ops.append(lin_op)
        total_length += lin_op.size[0]
    # Stack all the outputs into a single vector.
    return lu.vstack(vect_lin_ops, (total_length, 1))
