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

from cvxpy.atoms.affine.affine_atom import AffAtom
import cvxpy.utilities as u
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_utils as lu
from scipy.signal import fftconvolve

class conv2d(AffAtom):
    """ 2D discrete convolution of two matrices.
    """
    # TODO work with right hand constant.
    def __init__(self, lh_expr, rh_expr):
        super(conv2d, self).__init__(lh_expr, rh_expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Convolve the two values.
        """
        # First argument must be larger.
        if values[0].size >= values[1].size:
            return fftconvolve(values[0], values[1], mode='full')
        else:
            return fftconvolve(values[1], values[0], mode='full')

    def shape_from_args(self):
        """The sum of the argument dimensions - 1.
        """
        return u.Shape(self.args[0].size[0] + self.args[1].size[0] - 1,
                       self.args[0].size[1] + self.args[1].size[1] - 1)

    def sign_from_args(self):
        """Always unknown.
        """
        return u.Sign.UNKNOWN

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Convolve two vectors.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        return (lu.conv2d(arg_objs[0], arg_objs[1], size), [])
