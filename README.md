This is a special version of CVXPY designed for matrix-free convex optimization. See [these papers](http://stanford.edu/~boyd/papers/abs_ops.html) for further details on matrix-free convex optimization. See the [CVXPY site](http://www.cvxpy.org) for more information on CVXPY itself.

From the perspective of the user, matrix-free CVXPY is exactly the same as standard CVXPY except it has an additional solver "SCS_MAT_FREE", which is a matrix-free version of SCS.
The solver is preliminary and not at all optimized. 

To install matrix-free CVXPY, clone this Github repository and run ``python setup.py install`` in the repository directory.
You must also install the matrix-free SCS solver from source ([repository here](https://github.com/SteveDiamond/scs)).

See [this repository](https://github.com/SteveDiamond/FAO_DAG/blob/master/README.md) for instructions on installing a faster version of the matrix-free SCS solver.
