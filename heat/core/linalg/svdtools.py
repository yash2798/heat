"""
distributed hierarchical SVD - first draft
"""
import numpy as np
import collections
import torch
from typing import Type, Callable, Dict, Any, TypeVar, Union, Tuple

from ..communication import MPICommunication
from ..dndarray import DNDarray
from .. import factories
from .. import types
from ..linalg import matmul, vector_norm
from ..indexing import where
from ..random import randn

from ..manipulations import vstack, hstack, diag, balance
from ..exponential import sqrt

from .. import statistics
from math import log, ceil, floor


__all__ = ["hsvd"]


def hsvd(
    A: DNDarray,
    maxrank: int = 0,
    maxmergedim: int = 0,
    loctol: float = 5e-2,
    full: bool = False,
    silent: bool = True,
) -> Tuple[DNDarray, DNDarray, DNDarray]:
    """
    Computes an approximate truncated SVD of A utilizing a distributed hiearchical algorithm; the truncation rank is given by maxrank, i.e.
    if A = U diag(sigma) V^T is the true SVD of A, this routine computes an approximation for U[:,:maxrank] (and sigma[:maxrank], V[:,:maxrank]).

    WARNING: The results are quite accurate for an imput matrix with rank < maxrank, but can be highly unprecise for larger ranks.

    INPUT:
    A:          two-dimensional DNDarray; the matrix of which the approximate truncated SVD has to be computed
    maxrank:    integer; rank to which SVD is truncated.
    loctol:     ---currently not supported---
    full:       boolean; True: compute U[:,:maxrank], sigma[:maxrank], V[:,:maxrank], False: Compute only U (i.e. the raxrank leading left singular vectors)
    silent:     boolean; True: no information on the computational procedure is displayed, False: information is printed.

    OUTPUT:
    either U (if full=False) or U, sigma, V (if full=True); all of them as DNDarrays.

    input "loctol" is currently not supported. TODO: implement error control with local tolerance (instead of truncation rank) as in [2]...

    Selected References:
    [1] Iwen, Ong. A distributed and incremental SVD algorithm for agglomerative data analysis on large networks. SIAM J. Matrix Anal. Appl., 37(4), 2016.
    [2] Himpe, Leibner, Rave. Hierarchical approximate proper orthogonal decomposition. SIAM J. Sci. Comput., 40 (5), 2018.
    """
    if A.comm.rank == 0:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("WARNING: hierarchical SVD is intended for highly thin and/or low-")
        print("rank DNDarrays with split=1.")
        print("In other cases the result may be highly inaccurate!")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    if not isinstance(A, DNDarray):
        raise RuntimeError("Argument needs to be a DNDarray but is {}.".format(type(A)))
    if not A.ndim == 2:
        raise RuntimeError("A needs to be a 2D matrix")
    if not A.dtype == types.float32 and not A.dtype == types.float64:
        raise RuntimeError(
            "Argument needs to be a DNDarray with datatype float32 or float64, but data type is {}.".format(
                A.dtype
            )
        )

    # if split dimension is 0, transpose matrix and remember this
    transposeflag = False
    if A.split == 0:
        transposeflag = True
        A = A.T

    # if maxrank = 0 we choose a default value for maxrank
    # the chosen default value corresponds to merging along a binary tree (i.e. maxrank = local size / 2)
    if maxrank == 0:
        maxrank = floor(max(A.lshape_map[:, 1]) / 2)
    if maxmergedim == 0:
        maxmergedim = max(A.lshape_map[:, 1])
    # if the manually set maxrank is so large that with either the default or the manually set maxmergedim it would not be possible to merge (even along a binary tree), we increase maxmergedim and warn the user
    if 2 * maxrank > maxmergedim:
        maxmergedim = 2 * maxrank + 1
        if A.comm.rank == 0:
            print("WARNING: maxmergedim = 2*truncationrank +1 might cause memory issues!")

    no_procs = A.comm.Get_size()

    # compute the SVDs on the 0th level
    level = 0
    active_nodes = [i for i in range(no_procs)]
    if A.comm.rank == 0 and not silent:
        print("hSVD level %d..." % level, active_nodes)
    U_loc, sigma_loc, _ = torch.linalg.svd(A.larray, full_matrices=False)

    # truncation of the SVD at each "node"
    # *************************************************************************
    # !!! TODO:  implement adaptive truncation
    # *************************************************************************
    loc_trunc_rank = min(sigma_loc.shape[0], maxrank)
    U_loc = torch.linalg.matmul(U_loc[:, :loc_trunc_rank], torch.diag(sigma_loc[:loc_trunc_rank]))

    # if not silent:
    #    graph = np.ones(no_procs)

    finished = False
    while not finished:
        # communicate dimension of currenlty active nodes to all other nodes
        dims_global = [0] * no_procs
        dims_global[A.comm.rank] = U_loc.shape[1]
        for k in range(no_procs):
            dims_global[k] = A.comm.bcast(dims_global[k], root=k)

        # determine future nodes and prepare sending
        future_nodes = [0]
        send_to = [[]] * no_procs
        current_idx = 0
        current_future_node = 0
        used_budget = 0
        k = 0
        while k < len(active_nodes):
            current_idx = active_nodes[k]
            if used_budget + dims_global[current_idx] > maxmergedim:
                current_future_node = current_idx
                future_nodes.append(current_future_node)
                used_budget = dims_global[current_idx]
            else:
                if not used_budget == 0:
                    send_to[current_idx] = current_future_node
                used_budget += dims_global[current_idx]
            k += 1

        recv_from = [[]] * no_procs
        for i in future_nodes:
            recv_from[i] = [k for k in range(no_procs) if send_to[k] == i]

        # if not silent:
        #    graph = np.vstack([graph,
        #                       np.asarray([1 if i in future_nodes else 0 for i in range(no_procs)])])

        if A.comm.rank in future_nodes:
            # FUTURE NODES
            # in the future nodes receive local arrays from previous level
            U_loc = [U_loc] + [
                torch.zeros(
                    (A.shape[0], dims_global[i]), dtype=A.larray.dtype, device=A.device.torch_device
                )
                for i in recv_from[A.comm.rank]
            ]
            for k in range(len(recv_from[A.comm.rank])):
                A.comm.Recv(U_loc[k + 1], recv_from[A.comm.rank][k], tag=recv_from[A.comm.rank][k])
            # concatenate the received arrays
            U_loc = torch.hstack(U_loc)
            level += 1
            if A.comm.rank == 0 and not silent:
                print("hSVD level %d..." % level, "reduce to nodes", future_nodes)
            # compute "local" SVDs on the current level
            U_loc, sigma_loc, _ = torch.linalg.svd(U_loc, full_matrices=False)
            loc_trunc_rank = min(sigma_loc.shape[0], maxrank)
            # *****************************************************************
            # TODO: implement truncation
            # *****************************************************************
            if len(future_nodes) > 1:
                # prepare next level or...
                U_loc = torch.linalg.matmul(
                    U_loc[:, :loc_trunc_rank], torch.diag(sigma_loc[:loc_trunc_rank])
                )
            else:
                # or do only truncate in case of the final level
                U_loc = U_loc[:, :loc_trunc_rank]
        elif A.comm.rank in active_nodes and A.comm.rank not in future_nodes:
            # NOT FUTURE NODES
            # in these nodes we only send the local arrays to the respective future node
            A.comm.Send(U_loc, send_to[A.comm.rank], tag=A.comm.rank)

        if len(future_nodes) == 1:
            finished = True
        else:
            active_nodes = future_nodes

    # After completion of the SVD, distribute the result from process 0 to all processes again
    U_loc_shape = A.comm.bcast(U_loc.shape, root=0)
    if A.comm.rank != 0:
        U_loc = torch.zeros(U_loc_shape, dtype=A.larray.dtype, device=A.device.torch_device)
    req = A.comm.Ibcast(U_loc, root=0)
    req.Wait()

    U = factories.array(U_loc, device=A.device, split=None, comm=A.comm)

    # if not silent:
    #    np.savetxt('hsvd_graph.txt',graph)

    # Postprocessing:
    # compute V if required or if split=0 for the input
    # in case of split=0 undo the transposition...
    if transposeflag or full:
        V = matmul(A.T, U)
        sigma = vector_norm(V, axis=0)
        V = matmul(V, diag(1 / sigma))

        if transposeflag and full:
            return V, sigma, U
        elif transposeflag and not full:
            return V
        else:
            return U, sigma, V
        return U

    return U
