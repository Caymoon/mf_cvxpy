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

import cvxpy.settings as s
from cvxpy.problems.solvers.scs_intf import SCS
import cvxpy.problems.iterative as iterative
import cvxpy.lin_ops.tree_mat as tree_mat
import cvxpy.lin_ops.fao_utils as fao_utils
import mat_free_scs
import faoInterface

class MAT_FREE_SCS(SCS):
    """An interface for the SCS solver.
    """
    def name(self):
        """The name of the solver.
        """
        return s.MAT_FREE_SCS

    def split_constr(self, constr_map):
        """Extracts the equality, inequality, and nonlinear constraints.

        Parameters
        ----------
        constr_map : dict
            A dict of the canonicalized constraints.

        Returns
        -------
        tuple
            (eq_constr, ineq_constr, nonlin_constr)
        """
        return ([], [], [])

    def get_problem_data(self, objective, constraints, cached_data):
        """Returns the argument for the call to the solver.

        Parameters
        ----------
        objective : LinOp
            The canonicalized objective.
        constraints : list
            The list of canonicalized cosntraints.
        cached_data : dict
            A map of solver name to cached problem data.

        Returns
        -------
        tuple
            (solver args tuple, offset)
        """
        sym_data = self.get_sym_data(objective, constraints, cached_data)
        matrix_data = self.get_matrix_data(objective, constraints,
                                           cached_data)
        c, offset = matrix_data.get_objective()
        all_ineq = sym_data.constr_map[s.EQ] + sym_data.constr_map[s.LEQ]
        A_rows = sum(constr.size[0]*constr.size[1] for constr in all_ineq)
        constr_root = tree_mat.combine_lin_ops([con.expr for con in all_ineq])
        b = -tree_mat.mul(constr_root, {})
        data = {"c": c}
        data["b"] = b
        # Remove constants from expressions.
        tree_mat.prune_expr(constr_root)
        # Get ordered list of var sizes.
        var_offsets = sorted(sym_data.var_offsets.items(),
                             lambda x,y: x[0] <= y[0])
        ordered_sizes = [(v, sym_data.var_sizes[v]) for v,_ in var_offsets]
        data['fao_dag'] = fao_utils.tree_to_dag(constr_root, ordered_sizes)
        return (data, sym_data.dims), offset

    def solve(self, objective, constraints, cached_data, verbose, solver_opts):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        objective : LinOp
            The canonicalized objective.
        constraints : list
            The list of canonicalized cosntraints.
        cached_data : dict
            A map of solver name to cached problem data.
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        (data, dims), obj_offset = self.get_problem_data(objective,
                                                         constraints,
                                                         cached_data)
        results_dict = faoInterface.scs_solve(data['fao_dag'], data, dims, solver_opts)
        return self.format_results(results_dict, dims, obj_offset)

    def format_results(self, results_dict, dims, obj_offset=0):
        """Converts the solver output into standard form.

        Parameters
        ----------
        results_dict : dict
            The solver output.
        dims : dict
            The cone dimensions in the canonicalized problem.
        obj_offset : float, optional
            The constant term in the objective.

        Returns
        -------
        dict
            The solver output in standard form.
        """
        new_results = {}
        status = s.SOLVER_STATUS[s.SCS][results_dict["info"]["status"]]
        new_results[s.STATUS] = status
        new_results[s.SOLVE_TIME] = results_dict["info"]["solveTime"] + \
                                    results_dict["info"]["setupTime"]
        new_results['CG_ITERS'] = results_dict["info"]["cgIter"]
        if new_results[s.STATUS] in s.SOLUTION_PRESENT:
            primal_val = results_dict["info"]["pobj"]
            new_results[s.VALUE] = primal_val + obj_offset
            new_results[s.PRIMAL] = results_dict["x"]
            new_results[s.EQ_DUAL] = results_dict["y"][0:dims["f"]]
            new_results[s.INEQ_DUAL] = results_dict["y"][dims["f"]:]

        return new_results
