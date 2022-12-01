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

from .. import statistics
from math import log, ceil, floor, sqrt


__all__ = ["hsvd", "hpod"]


def hsvd(
    A: DNDarray,
    maxrank: Union[int, None] = None,
    maxmergedim: Union[int, None] = None,
    reltol: Union[float, None] = None,
    safetyshift: int = 0,
    no_of_merges: Union[int, None] = None,
    full: bool = False,
    silent: bool = True,
) -> Union[
    Tuple[DNDarray, DNDarray, DNDarray, float], Tuple[DNDarray, DNDarray, DNDarray], DNDarray
]:
    """
    Computes an approximate truncated SVD of A utilizing a distributed hiearchical algorithm; the truncation rank is given by maxrank, i.e.
    if A = U diag(sigma) V^T is the true SVD of A, this routine computes an approximation for U[:,:maxrank] (and sigma[:maxrank], V[:,:maxrank]).

    WARNING: The results are quite accurate for an imput matrix with rank < maxrank, but can be highly unprecise for larger ranks.

    INPUT:
    A:          two-dimensional DNDarray; the matrix of which the approximate truncated SVD has to be computed
    maxrank:    integer; rank to which SVD is truncated.
    reltol:     tolerance for the relative error ||A-U Sigma V^T ||_F / ||A||_F (computed according to [2] for the "worst-case" of merging along a binary tree)
    full:       boolean; True: compute U[:,:maxrank], sigma[:maxrank], V[:,:maxrank], False: Compute only U (i.e. the raxrank leading left singular vectors)
    silent:     boolean; True: no information on the computational procedure is displayed, False: information is printed.
    safetyshift:    increases the truncation rank by adding a safety shift.

    OUTPUT:
    either U (if full=False) or U, sigma, V (if full=True); all of them as DNDarrays.
    Moreover, an upper estimate (rel_error_estimate) for ||A-U Sigma V^T ||_F / ||A||_F (computed according to [2] along the "true" merging tree) is returned.

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

    # Choice of the parameters
    # A_local_size = max(A.lshape_map[:, 1])
    no_procs = A.comm.Get_size()

    Anorm = vector_norm(A)

    if reltol is not None:
        loctol = Anorm.larray * reltol / sqrt(2 * no_procs - 1)
    else:
        loctol = None

    # compute the SVDs on the 0th level
    level = 0
    active_nodes = [i for i in range(no_procs)]
    if A.comm.rank == 0 and not silent:
        print("hSVD level %d..." % level, active_nodes)

    # U_loc, sigma_loc, _ = torch.linalg.svd(A.larray, full_matrices=False)
    # if reltol is None:
    #     loc_trunc_rank = min(sigma_loc.shape[0], maxrank)
    # else:
    #     ideal_trunc_rank = min(
    #         torch.argwhere(
    #             torch.tensor(
    #                 [torch.norm(sigma_loc[k:]) ** 2 for k in range(sigma_loc.shape[0] + 1)]
    #             )
    #             < loctol**2
    #         )
    #     )
    #     loc_trunc_rank = min(maxrank, ideal_trunc_rank)
    #     if loc_trunc_rank != ideal_trunc_rank:
    #         print(
    #             "Warning (level %d, process %d): reltol = %2.2e requires truncation to rank %d, but maxrank=%d. Possible loss of desired precision."
    #             % (level, A.comm.rank, reltol, ideal_trunc_rank, maxrank)
    #         )
    # U_loc = torch.linalg.matmul(U_loc[:, :loc_trunc_rank], torch.diag(sigma_loc[:loc_trunc_rank]))
    # err_squared_loc = torch.linalg.norm(sigma_loc[loc_trunc_rank:]) ** 2

    U_loc, sigma_loc, err_squared_loc = compute_local_truncated_svd(
        level, A.comm.rank, A.larray, maxrank, loctol, safetyshift
    )
    U_loc = torch.linalg.matmul(U_loc, torch.diag(sigma_loc))

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
        counter = 0
        while k < len(active_nodes):
            current_idx = active_nodes[k]
            if used_budget + dims_global[current_idx] > maxmergedim or counter == no_of_merges:
                current_future_node = current_idx
                future_nodes.append(current_future_node)
                used_budget = dims_global[current_idx]
                counter = 1
            else:
                if not used_budget == 0:
                    send_to[current_idx] = current_future_node
                used_budget += dims_global[current_idx]
                counter += 1
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
            err_squared_loc = [err_squared_loc] + [
                torch.zeros_like(err_squared_loc) for i in recv_from[A.comm.rank]
            ]
            U_loc = [U_loc] + [
                torch.zeros(
                    (A.shape[0], dims_global[i]), dtype=A.larray.dtype, device=A.device.torch_device
                )
                for i in recv_from[A.comm.rank]
            ]
            for k in range(len(recv_from[A.comm.rank])):
                A.comm.Recv(U_loc[k + 1], recv_from[A.comm.rank][k], tag=recv_from[A.comm.rank][k])
                A.comm.Recv(
                    err_squared_loc[k + 1],
                    recv_from[A.comm.rank][k],
                    tag=2 * no_procs + recv_from[A.comm.rank][k],
                )
            # concatenate the received arrays
            U_loc = torch.hstack(U_loc)
            err_squared_loc = sum(err_squared_loc)
            level += 1
            if A.comm.rank == 0 and not silent:
                print("hSVD level %d..." % level, "reduce to nodes", future_nodes)
            # compute "local" SVDs on the current level

            if len(future_nodes) == 1:
                safetyshift = 0
            U_loc, sigma_loc, err_squared_loc_new = compute_local_truncated_svd(
                level, A.comm.rank, U_loc, maxrank, loctol, safetyshift
            )

            # U_loc, sigma_loc, _ = torch.linalg.svd(U_loc, full_matrices=False)
            # if reltol is None:
            #     loc_trunc_rank = min(sigma_loc.shape[0], maxrank)
            # else:
            #     ideal_trunc_rank = min(
            #         torch.argwhere(
            #             torch.tensor(
            #                 [torch.norm(sigma_loc[k:]) ** 2 for k in range(sigma_loc.shape[0] + 1)]
            #             )
            #             < loctol**2
            #         )
            #     )
            #     loc_trunc_rank = min(maxrank, ideal_trunc_rank)
            #     if loc_trunc_rank != ideal_trunc_rank:
            #         print(
            #             "Warning (level %d, process %d): reltol = %2.2e requires truncation to rank %d, but maxrank=%d. Possible loss of desired precision."
            #             % (level, A.comm.rank, reltol, ideal_trunc_rank, maxrank)
            #         )

            if len(future_nodes) > 1:
                # prepare next level or...
                U_loc = torch.linalg.matmul(U_loc, torch.diag(sigma_loc))
            err_squared_loc += err_squared_loc_new
        elif A.comm.rank in active_nodes and A.comm.rank not in future_nodes:
            # NOT FUTURE NODES
            # in these nodes we only send the local arrays to the respective future node
            A.comm.Send(U_loc, send_to[A.comm.rank], tag=A.comm.rank)
            A.comm.Send(err_squared_loc, send_to[A.comm.rank], tag=2 * no_procs + A.comm.rank)

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
    req = A.comm.Ibcast(err_squared_loc, root=0)
    req.Wait()

    U = factories.array(U_loc, device=A.device, split=None, comm=A.comm)

    rel_error_estimate = (
        factories.array(err_squared_loc**0.5, device=A.device, split=None, comm=A.comm) / Anorm
    )

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
            return V, sigma, U, rel_error_estimate
        elif transposeflag and not full:
            return V, rel_error_estimate
        else:
            return U, sigma, V, rel_error_estimate
        return U, rel_error_estimate

    return U, rel_error_estimate


def hpod(
    A: DNDarray,
    weight_matrix: Union[torch.Tensor, None] = None,
    maxrank: Union[int, None] = None,
    maxmergedim: Union[int, None] = None,
    silent: bool = True,
) -> Tuple[DNDarray, float]:
    """
    Computes the POD of rank r=maxrank for the snapshot set given by the column vectors of A.
    The weight matrix for the scalar product is given by weight_matrix.

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Prelimiary and experimental!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Selected References:
    [1] Himpe, Leibner, Rave. Hierarchical approximate proper orthogonal decomposition. SIAM J. Sci. Comput., 40 (5), 2018.
    """
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
    if weight_matrix is not None and weight_matrix.dtype != A.dtype.torch_type():
        raise RuntimeError(
            "Data types of input A and input weight_matrix need to coicide, but are {} and {}.".format(
                A.dtype.torch_type(), weight_matrix.dtype
            )
        )

    if A.split == 0:
        raise RuntimeError(
            "Argument A needs to have split dimension 1. Consider, e.g., calling A.resplit_(1) before using hpod."
        )

    # if maxrank = 0 we choose a default value for maxrank
    # the chosen default value corresponds to merging along a binary tree (i.e. maxrank = local size / 2)
    if maxrank is None:
        maxrank = floor(max(A.lshape_map[:, 1]) / 2)
    if maxmergedim is None:
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
        print("hPOD level %d..." % level, active_nodes)

    U_loc, sigma_loc, err_squared_loc = compute_local_pod(
        level, A.comm.rank, A.larray, maxrank, None, weight_matrix
    )
    U_loc = torch.linalg.matmul(U_loc, torch.diag(sigma_loc))

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
            err_squared_loc = [err_squared_loc] + [
                torch.zeros_like(err_squared_loc) for i in recv_from[A.comm.rank]
            ]
            U_loc = [U_loc] + [
                torch.zeros(
                    (A.shape[0], dims_global[i]), dtype=A.larray.dtype, device=A.device.torch_device
                )
                for i in recv_from[A.comm.rank]
            ]
            for k in range(len(recv_from[A.comm.rank])):
                A.comm.Recv(U_loc[k + 1], recv_from[A.comm.rank][k], tag=recv_from[A.comm.rank][k])
                A.comm.Recv(
                    err_squared_loc[k + 1],
                    recv_from[A.comm.rank][k],
                    tag=2 * no_procs + recv_from[A.comm.rank][k],
                )
            # concatenate the received arrays
            U_loc = torch.hstack(U_loc)
            err_squared_loc = sum(err_squared_loc)
            level += 1
            if A.comm.rank == 0 and not silent:
                print("hPOD level %d..." % level, "reduce to nodes", future_nodes)
            # compute "local" SVDs on the current level

            U_loc, sigma_loc, err_squared_loc_new = compute_local_pod(
                level, A.comm.rank, U_loc, maxrank, None, weight_matrix
            )

            if len(future_nodes) > 1:
                # prepare next level or...
                U_loc = torch.linalg.matmul(U_loc, torch.diag(sigma_loc))
            err_squared_loc += err_squared_loc_new
        elif A.comm.rank in active_nodes and A.comm.rank not in future_nodes:
            # NOT FUTURE NODES
            # in these nodes we only send the local arrays to the respective future node
            A.comm.Send(U_loc, send_to[A.comm.rank], tag=A.comm.rank)
            A.comm.Send(err_squared_loc, send_to[A.comm.rank], tag=2 * no_procs + A.comm.rank)

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
    req = A.comm.Ibcast(err_squared_loc, root=0)
    req.Wait()

    U = factories.array(U_loc, device=A.device, split=None, comm=A.comm)

    mean_sq_error_estimate = (
        factories.array(err_squared_loc, device=A.device, split=None, comm=A.comm) / A.shape[1]
    )

    # if not silent:
    #    np.savetxt('hsvd_graph.txt',graph)

    return U, mean_sq_error_estimate


def compute_local_truncated_svd(
    level: int,
    proc_id: int,
    U_loc: torch.Tensor,
    maxrank: int,
    loctol: Union[float, None],
    safetyshift: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Auxiliary routine for hsvd: computes the truncated SVD of the respective local array
    """
    U_loc, sigma_loc, _ = torch.linalg.svd(U_loc, full_matrices=False)

    if U_loc.dtype == torch.float64:
        noiselevel = 1e-14
    elif U_loc.dtype == torch.float32:
        noiselevel = 1e-7

    cut_noise_rank = max(torch.argwhere(sigma_loc >= noiselevel)) + 1

    if loctol is None:
        loc_trunc_rank = min(maxrank, cut_noise_rank)
    else:
        ideal_trunc_rank = min(
            torch.argwhere(
                torch.tensor(
                    [torch.norm(sigma_loc[k:]) ** 2 for k in range(sigma_loc.shape[0] + 1)],
                    device=U_loc.device,
                )
                < loctol**2
            )
        )
        loc_trunc_rank = min(maxrank, ideal_trunc_rank, cut_noise_rank)
        if loc_trunc_rank != ideal_trunc_rank:
            print(
                "Warning (level %d, process %d): abs tol = %2.2e requires truncation to rank %d, but maxrank=%d. Possible loss of desired precision."
                % (level, proc_id, loctol, ideal_trunc_rank, maxrank)
            )

    loc_trunc_rank = min(sigma_loc.shape[0], loc_trunc_rank + safetyshift)
    err_squared_loc = torch.linalg.norm(sigma_loc[loc_trunc_rank - safetyshift :]) ** 2
    return U_loc[:, :loc_trunc_rank], sigma_loc[:loc_trunc_rank], err_squared_loc


def compute_local_pod(
    level: int,
    proc_id: int,
    U_loc: torch.Tensor,
    maxrank: int,
    loctol: Union[float, None],
    weight: Union[torch.Tensor, None],
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Auxiliary routine for hpod: computes the POD of the respective local array
    """
    if weight is None:
        U_loc, sigma_loc, _ = torch.linalg.svd(U_loc, full_matrices=False)

        if loctol is None:
            loc_trunc_rank = min(sigma_loc.shape[0], maxrank)
        else:
            ideal_trunc_rank = min(
                torch.argwhere(
                    torch.tensor(
                        [torch.norm(sigma_loc[k:]) ** 2 for k in range(sigma_loc.shape[0] + 1)],
                        device=U_loc.device,
                    )
                    < loctol**2
                )
            )
            loc_trunc_rank = min(maxrank, ideal_trunc_rank)
            if loc_trunc_rank != ideal_trunc_rank:
                print(
                    "Warning (level %d, process %d): abs tol = %2.2e requires truncation to rank %d, but maxrank=%d. Possible loss of desired precision."
                    % (level, proc_id, loctol, ideal_trunc_rank, maxrank)
                )
        err_squared_loc = torch.linalg.norm(sigma_loc[loc_trunc_rank:]) ** 2
        return U_loc[:, :loc_trunc_rank], sigma_loc[:loc_trunc_rank], err_squared_loc

    if weight is not None:
        gramian = U_loc.T @ weight @ U_loc
        lamda, V = torch.linalg.eigh(gramian)
        gramian_trace = torch.trace(gramian)

        lamda = torch.flip(lamda, [0])
        V = torch.flip(V, [1])

        if U_loc.dtype == torch.float64:
            noiselevel = 1e-14
        elif U_loc.dtype == torch.float32:
            noiselevel = 1e-7

        cut_noise_rank = min(U_loc.shape[0], max(torch.argwhere(lamda > noiselevel)))

        if loctol is None:
            ideal_trunc_rank = maxrank
        else:
            # ideal_trunc_rank = min(
            #     torch.argwhere(
            #         torch.tensor(
            #             [sum(lamda[:k]) for k in range(lamda.shape[0] + 1)]
            #         )
            #         > gramian_trace - loctol**2
            #     )
            # )
            ideal_trunc_rank = min(
                torch.argwhere(
                    torch.tensor(
                        [torch.sum(lamda[k:]) for k in range(lamda.shape[0])], device=U_loc.device
                    )
                    < loctol**2
                )
            )
            if ideal_trunc_rank > maxrank:
                print(
                    "Warning (level %d, process %d): desired abs tol = %2.2e would require truncation to rank %d, but maxrank=%d. Possible loss of desired precision."
                    % (level, proc_id, loctol, ideal_trunc_rank, maxrank)
                )
        if ideal_trunc_rank > cut_noise_rank:
            print(
                "Warning (level %d, process %d): truncation to rank %d is intended, but numerical noise of eigh starts after rank %d. Possible loss of desired precision."
                % (level, proc_id, ideal_trunc_rank, cut_noise_rank)
            )
        loc_trunc_rank = min(maxrank, cut_noise_rank, ideal_trunc_rank)
        err_squared_loc = gramian_trace - torch.sum(lamda[:loc_trunc_rank])
        sigma = lamda[:loc_trunc_rank] ** 0.5
        U = U_loc @ V[:, :loc_trunc_rank] @ torch.diag(1.0 / sigma)
        return U, sigma, err_squared_loc
