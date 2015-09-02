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

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from collections import namedtuple
import numpy as np
from collections import deque, defaultdict

# A forward adjoint oracle.
# type: The FAO type.
# input_sizes: A list of tuples with the dimensions of the inputs.
# output_sizes: A list of tuples with the dimensions of the inputs.
# input_nodes: A list of FAO inputs.
# output_nodes: A list of FAO outputs.
# data: Other information needed by the FAO.
FAO = namedtuple("FAO", ["type",
                         "input_sizes",
                         "output_sizes",
                         "input_edges",
                         "output_edges",
                         "data"])

# A DAG of FAOs.
# start_node: The DAG start node.
# end_node: The DAG end node.
# nodes: Map of node id to node.
# edges: Map of edge id to edge.
FAO_DAG = namedtuple("FAO_DAG", ["start_node",
                                 "end_node",
                                 "nodes",
                                 "edges"])

# FAO types:

# Multiplying an expression by a scalar constant.
# Data: The scalar constant.
SCALAR_MUL = "scalar_mul"
# Multiplying a dense matrix constant by a vector.
# Data: The matrix constant.
DENSE_MAT_VEC_MUL = "dense_mat_vec_mul"
# Multiplying a dense matrix constant by a matrix.
# Data: The matrix constant.
DENSE_MAT_MAT_MUL = "dense_mat_mat_mul"
# Multiplying a sparse matrix constant by a vector.
# Data: The matrix constant.
SPARSE_MAT_VEC_MUL = "sparse_mat_vec_mul"
# Multiplying a sparse matrix constant by a matrix.
# Data: The matrix constant.
SPARSE_MAT_MAT_MUL = "sparse_mat_mat_mul"
# Copies an input onto multiple outputs.
# Data: None.
COPY = "copy"
# Splits an input into multiple outputs.
# Data: None.
SPLIT = "split"

def lin_op_data(lin_op):
    """Return the LinOp data converted into FAO format.
    """
    fao_data = lin_op.data.data
    if isinstance(fao_data, np.matrix):
        fao_data = fao_data.A
    return fao_data

def lin_op_to_fao(root, edges):
    """Converts a LinOp tree to an FAO tree.

    Parameters
    ----------
    root: The root of a LinOp tree.
    edges: A list of (FAO, FAO) tuples.

    Returns
    -------
    The root of an FAO tree.
    """
    # Split multiplication into different types.
    if root.type == lo.MUL:
        if root.data.type == lo.SCALAR_CONST:
            fao_type = SCALAR_MUL
        elif root.data.type == lo.DENSE_CONST:
            if root.size[1] == 1:
                fao_type = DENSE_MAT_VEC_MUL
            else:
                fao_type = DENSE_MAT_MAT_MUL
        else:
            if root.size[1] == 1:
                fao_type = SPARSE_MAT_VEC_MUL
            else:
                fao_type = SPARSE_MAT_MAT_MUL
        fao_data = lin_op_data(root)
    # Convert division to scalar multiplication.
    elif root.type == lo.DIV:
        fao_type = SCALAR_MUL
        fao_data = 1.0/root.data.data
    # Conv data is the kernel.
    elif root.type == lo.CONV:
        fao_type = lo.CONV
        fao_data = lin_op_data(root).flatten()
    else:
        fao_type = root.type
        fao_data = root.data
    root_fao = FAO(fao_type, [], [root.size], [], [], fao_data)
    # Convert arg subtrees to FAO trees
    # and fill out FAO data.
    for arg in root.args:
        arg_fao = lin_op_to_fao(arg, edges)
        edge_id = new_edge(arg_fao, root_fao, edges)
        arg_fao.output_edges.append(edge_id)
        root_fao.input_sizes.append(arg.size)
        root_fao.input_edges.append(edge_id)
    return root_fao

def get_leaves(root, edges, var_nodes, no_op_nodes):
    """Adds all the variables and NO OPs to the appropriate lists.
    """
    if root.type == lo.VARIABLE:
        var_nodes.append(root)
    elif root.type == lo.NO_OP:
        no_op_nodes.append(root)

    for edge_idx in root.input_edges:
        input_node = edges[edge_idx][0]
        get_leaves(input_node, edges, var_nodes, no_op_nodes)

def simplify_dag(dag):
    """Eliminates unnecessary nodes in the DAG.

    Parameters
    ----------
    dag: An FAO DAG.

    Returns
    -------
    An FAO DAG.
    """
    edges = dag.edges.copy()
    start_node = dag.start_node
    end_node = dag.end_node
    # Traverse the graph. Eliminate any identity transformations,
    # e.g, + with one arg, copy with one output.
    eval_map = defaultdict(int)
    ready_queue = deque()
    ready_queue.append(start_node)
    while len(ready_queue) > 0:
        curr = ready_queue.popleft()
        eval_map[id(curr)] += 1
        # Last node visited is end node.
        end_node = curr
        # TODO very specific.
        # Rewrite f(-,...,-) as -f(...) if more inputs than outputs.
        if all_inputs_neg(curr, edges) and \
            len(curr.output_edges) < len(curr.input_edges):
            for idx, edge_idx in enumerate(curr.input_edges):
                # Replace input edge with neg input edge.
                neg_input = edges[edge_idx][0]
                del edges[edge_idx]
                new_edge_id = neg_input.input_edges[0]
                curr.input_edges[idx] = new_edge_id
                edges[new_edge_id] = (edges[new_edge_id][0], curr)
            # Create new NEG nodes as outputs.
            # Special case for end node.
            if len(curr.output_edges) == 0:
                curr.output_edges.append(None)
            for idx, edge_idx in enumerate(curr.output_edges):
                size = curr.output_sizes[idx]
                # Special case for end node.
                if edge_idx is None:
                    outputs = []
                else:
                    outputs = [edge_idx]
                neg_node = FAO(lo.NEG, [size], [size], [], outputs, None)
                new_edge_id = new_edge(curr, neg_node, edges)
                neg_node.input_edges.append(new_edge_id)
                curr.output_edges[idx] = new_edge_id
        # If each input has visited the node, it is ready.
        for edge_idx in curr.output_edges:
            edge = edges[edge_idx]
            node = edge[1]
            # If this is an identity transformation, eliminate it.
            # It's a while loop in case the next node can also be eliminated.
            while True:
                if (node.type == COPY and len(node.output_edges) == 1) or \
                   (node.type == lo.SUM and len(node.input_edges) == 1):
                    edge_to_next_idx = node.output_edges[0]
                    next_node = edges[edge_to_next_idx][1]
                    new_edge_id = new_edge(curr, next_node, edges)
                    # Link curr and next_node.
                    edge_pos = curr.output_edges.index(edge_idx)
                    curr.output_edges[edge_pos] = new_edge_id
                    edge_pos = next_node.input_edges.index(edge_to_next_idx)
                    next_node.input_edges[edge_pos] = new_edge_id
                    del edges[node.output_edges[0]]
                    del edges[edge_idx]
                    node = next_node
                    edge_idx = new_edge_id
                else:
                    break
            eval_map[id(node)] += 1
            node_inputs_count = len(node.input_edges)
            if (eval_map[id(node)] == node_inputs_count):
                ready_queue.append(node)

    # If the starting split node only has one output,
    # eliminate it.
    if len(start_node.output_edges) == 1:
        edge_idx = start_node.output_edges[0]
        start_node = edges[edge_idx][1]
        del edges[edge_idx]
        start_node.input_edges[:] = []

    nodes = get_nodes(edges, start_node)
    return FAO_DAG(start_node, end_node, nodes, edges)

def all_inputs_neg(node, edges):
    """Are all the inputs to the node NEG nodes?
    """
    for edge_idx in node.input_edges:
        if edges[edge_idx][0].type != lo.NEG:
            return False
    return True

def get_nodes(edges, start_node):
    """Returns all the nodes accessible via the edges.

    Include the start node in case no edges.
    """
    nodes = {id(start_node): start_node}
    for n1, n2 in edges.values():
        nodes[id(n1)] = n1
        nodes[id(n2)] = n2
    return nodes

def new_edge(start_node, end_node, edges):
    """Create a new edge and add it to edges.
    """
    edge_id = lu.get_id()
    edges[edge_id] = (start_node, end_node)
    return edge_id

def tree_to_dag(root, ordered_vars):
    """Converts a LinOp tree into an FAO DAG.

    The start node is a split node, which splits a single input into
    the variables in the order given.

    Parameters
    ----------
    root: The root of a LinOp tree.
    ordered_vars: A list of (var id, var size) tuples.

    Returns
    -------
    tuple
        The (start, end) nodes of an FAO DAG.
    """
    edges = {}
    root = lin_op_to_fao(root, edges)
    input_len = sum([s[0]*s[1] for _,s in ordered_vars])
    start_node = FAO(SPLIT, [(input_len, 1)], [], [], [], None)
    # Get the variables and no_ops.
    var_nodes = []
    no_op_nodes = []
    get_leaves(root, edges, var_nodes, no_op_nodes)
    # Create a copy node for each variable.
    var_copies = {}
    for var_id, size in ordered_vars:
        copy_node = FAO(COPY, [size], [], [], [], None)
        edge_id = new_edge(start_node, copy_node, edges)
        copy_node.input_edges.append(edge_id)
        start_node.output_sizes.append(size)
        start_node.output_edges.append(edge_id)
        var_copies[var_id] = copy_node
    # Link copy nodes directly to outputs of variables.
    for var in var_nodes:
        copy_node = var_copies[var.data]
        output_edge_id = var.output_edges[0]
        output_node = edges[output_edge_id][1]
        copy_node.output_sizes.append(var.output_sizes[0])
        edge_id = new_edge(copy_node, output_node, edges)
        copy_node.output_edges.append(edge_id)
        idx = output_node.input_edges.index(output_edge_id)
        output_node.input_edges[idx] = edge_id
        del edges[output_edge_id]
    # Link a copy node to all the NO_OPs.
    copy_node = var_copies[ordered_vars[0][0]]
    var_size = ordered_vars[0][1]
    for no_op_node in no_op_nodes:
        copy_node.output_sizes.append(var_size)
        edge_id = new_edge(copy_node, no_op_node, edges)
        copy_node.output_edges.append(edge_id)
        no_op_node.input_sizes.append(var_size)
        no_op_node.input_edges.append(edge_id)

    nodes = get_nodes(edges, start_node)
    dag = FAO_DAG(start_node, root, nodes, edges)
    return simplify_dag(dag)
