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
import canonInterface

class POGS(SCS):
    """An interface for the POGS solver.
    """
    def name(self):
        """The name of the solver.
        """
        return s.POGS

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
        cones = self.convert_dims_to_indices(dims)
        # Set the options to be VERBOSE plus any user-specific options.
        solver_opts["verbose"] = verbose
        results_dict = canonInterface.pogs_solve(data, cones)
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
        new_results[s.SOLVE_TIME] = (results_dict["info"]["solveTime"] + \
                                     results_dict["info"]["setupTime"])/1000.
        if new_results[s.STATUS] in s.SOLUTION_PRESENT:
            primal_val = results_dict["info"]["pobj"]
            new_results[s.VALUE] = primal_val + obj_offset
            new_results[s.PRIMAL] = results_dict["x"]
            new_results[s.EQ_DUAL] = results_dict["y"][0:dims["f"]]
            new_results[s.INEQ_DUAL] = results_dict["y"][dims["f"]:]

        return new_results
