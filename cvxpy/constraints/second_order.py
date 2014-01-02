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

class SOC(object):
    """A second-order cone constraint, i.e., norm2(x) <= t.

    Attributes:
        t: The scalar part of the second-order constraint.
        x_elems: The elements of the vector part of the constraint.
    """
    def __init__(self, t, x_elems):
        self.t = t
        self.x_elems = x_elems
        super(SOC, self).__init__()

    def __str__(self):
        return "SOC(%s, %s)" % (self.t, self.x_elems)

    # Formats SOC constraints for the solver.
    def format(self):
        constraints = [0 <= self.t]
        for elem in self.x_elems:
            constraints.append(0 <= elem)
        return constraints

    # The dimensions of the second-order cone.
    @property
    def size(self):
        rows = 1
        for elem in self.x_elems:
            rows += elem.size[0]*elem.size[1]
        return (rows, 1)
