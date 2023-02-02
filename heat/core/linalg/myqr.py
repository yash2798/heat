"""
Some auxiliary stuff for my programming... whatch out: not tested etc.!
"""
import numpy as np
import collections
import torch
from typing import Type, Callable, Dict, Any, TypeVar, Union, Tuple, Sequence, Optional, List

from ..communication import MPICommunication, Communication
from ..dndarray import DNDarray
from .. import factories
from .. import types
from ..linalg import *
from ..indexing import where
from ..random import randn
from ..devices import Device

from ..manipulations import vstack, hstack, diag, balance
from ..exponential import sqrt

from .. import statistics

from mpi4py import MPI


__all__ = ["myqr", "triu_solver", "myrandn"]


def myqr(
    A: DNDarray, calc_R: bool = False, full_Q: bool = False, crop_R_at_the_end: int = 0
) -> Union[DNDarray, Tuple[DNDarray, DNDarray]]:
    """
    My qr decomposition based on Block Gram Schmidt
    """
    if not isinstance(A, DNDarray):
        raise RuntimeError("Argument needs to be a DNDarray but is {}.".format(type(A)))
    if not A.ndim == 2:
        raise RuntimeError("A needs to be a 2D matrix")
    if not A.split == 1 and A.split is not None:
        raise RuntimeError(
            "Split dimension of input array must be 1 or None, but is {}.".format(A.split)
        )

    if A.split is None:
        if not full_Q:
            Q, R = torch.linalg.qr(A.larray, mode="reduced")
        else:
            Q, R = torch.linalg.qr(A.larray)
        Q = factories.array(Q, dtype=A.dtype, split=None, device=A.device, comm=A.comm)
        if calc_R:
            R = factories.array(R, dtype=A.dtype, split=None, device=A.device, comm=A.comm)
            return Q, R
        else:
            return Q

    if A.split == 1:

        if full_Q:
            if A.shape[1] < A.shape[0]:
                fill_up_array = myrandn(
                    (A.shape[0], A.shape[0] - A.shape[1]),
                    dtype=A.dtype,
                    split=A.split,
                    device=A.device,
                    comm=A.comm,
                )
                A_tilde = hstack([A, fill_up_array]).balance()
                # A_tilde = hstack([A, factories.ones((A.shape[0], A.shape[0]-A.shape[1]), dtype=A.dtype, split=A.split,device=A.device, comm=A.comm)]).balance()
                return myqr(A_tilde, calc_R=calc_R, full_Q=full_Q, crop_R_at_the_end=A.shape[1])

        lshapes = A.lshape_map[:, 1]
        lshapes_cum = torch.cumsum(lshapes, 0)
        nprocs = A.comm.size

        if A.shape[0] >= A.shape[1]:
            last_row_reached = nprocs
            k = A.shape[1]
        else:
            last_row_reached = min(torch.argwhere(lshapes_cum >= A.shape[0]))[0]
            k = A.shape[0]

        Q = factories.zeros(A.shape, dtype=A.dtype, split=1, device=A.device, comm=A.comm)
        if calc_R:
            R = factories.zeros(
                (k, A.shape[1]), dtype=A.dtype, split=1, device=A.device, comm=A.comm
            )
            R_shapes = torch.hstack(
                [torch.zeros(1, dtype=torch.int32), torch.cumsum(R.lshape_map[:, 1], 0)]
            )
        A_columns = A.larray.clone()

        for i in range(last_row_reached + 1):

            if i < nprocs - 1:
                k_loc_i = min(A.shape[0], A.lshape_map[i, 1])
                Q_buf = torch.zeros(
                    (A.shape[0], k_loc_i), dtype=A.larray.dtype, device=A.device.torch_device
                )

            if A.comm.rank == i:
                Q.larray, R_loc = torch.linalg.qr(A_columns, mode="reduced")
                if i < nprocs - 1:
                    Q_buf = Q.larray
                if calc_R:
                    r_size = R.larray[R_shapes[i] : R_shapes[i + 1], :].shape[0]
                    R.larray[R_shapes[i] : R_shapes[i + 1], :] = R_loc[:r_size, :]

            if i < nprocs - 1:
                req = A.comm.Ibcast(Q_buf, root=i)
                req.Wait()

            if A.comm.rank > i:
                R_loc = Q_buf.T @ A_columns
                A_columns -= Q_buf @ R_loc
                if calc_R:
                    r_size = R.larray[R_shapes[i] : R_shapes[i + 1], :].shape[0]
                    R.larray[R_shapes[i] : R_shapes[i + 1], :] = R_loc[:r_size, :]

        if calc_R:
            if crop_R_at_the_end != 0:
                return Q[:, :k].balance(), R[:, :crop_R_at_the_end].balance()
            else:
                return Q[:, :k].balance(), R
        else:
            return Q[:, :k].balance()

    elif A.split == 0:

        raise NotImplementedError("Not yet implemented.")


# -----------------------------------------------------------------------------
# old version with serial sends instead of broadcast...

# def myqr_old(A: DNDarray) -> DNDarray:

#     if not isinstance(A, DNDarray):
#         raise RuntimeError("Argument needs to be a DNDarray but is {}.".format(type(A)))
#     if not A.ndim == 2:
#         raise RuntimeError("A needs to be a 2D matrix")
#     if not A.split == 1 and A.split is not None:
#         raise RuntimeError(
#             "Split dimension of input array must be 1 or None, but is {}.".format(A.split)
#         )

#     if A.split is None:
#         Q, R = torch.linalg.qr(A.larray, mode="reduced")
#         Q = factories.array(Q, dtype=A.dtype, split=None, device=A.device, comm=A.comm)
#         return Q

#     lshapes = A.lshape_map[:, 1]
#     lshapes_cum = torch.cumsum(lshapes, 0)
#     nprocs = A.comm.size

#     if A.shape[0] >= A.shape[1]:
#         last_row_reached = nprocs
#         k = A.shape[1]
#     else:
#         last_row_reached = min(torch.argwhere(lshapes_cum >= A.shape[0]))[0]
#         k = A.shape[0]

#     Q = factories.zeros(A.shape, dtype=A.dtype, split=1, device=A.device, comm=A.comm)
#     A_columns = A.larray

#     for i in range(last_row_reached + 1):

#         if i < nprocs - 1:
#             k_loc_i = min(A.shape[0], A.lshape_map[i, 1])

#         if A.comm.rank == i:
#             Q.larray, _ = torch.linalg.qr(A_columns, mode="reduced")

#             snd_reqs = [0] * (nprocs - i - 1)
#             for j in range(i + 1, nprocs):
#                 snd_reqs[j - i - 1] = A.comm.Isend(Q.larray, j, tag=i * j)
#             [req.Wait() for req in snd_reqs]

#         elif A.comm.rank > i:
#             Q_from_i = torch.zeros(
#                 (A.shape[0], k_loc_i), dtype=A.larray.dtype, device=A.device.torch_device
#             )
#             A.comm.Recv(Q_from_i, i, tag=A.comm.rank * i)
#             R_loc = Q_from_i.T @ A_columns
#             A_columns -= Q_from_i @ R_loc

#     return Q[:, :k].balance()


def triu_solver(A: DNDarray, b: DNDarray) -> DNDarray:
    """
    My triangular solver, based on blockwise trisolves...
    """
    if not isinstance(A, DNDarray) or not isinstance(b, DNDarray):
        raise RuntimeError("Arguments need to be a DNDarrays.")
    if not A.ndim == 2:
        raise RuntimeError("A needs to be a 2D matrix")
    if not b.ndim <= 2:
        raise RuntimeError("b needs to be a vector (1D) or a matrix (2D)")
    if not A.shape[0] == A.shape[1]:
        raise RuntimeError("A needs to be a square matrix.")
    if not (b.split == 0 or b.split is None):
        raise RuntimeError("split=1 is not allowed for the right hand side.")
    if not b.shape[0] == A.shape[0]:
        raise RuntimeError("Dimension mismath of A and b.")
    if (
        A.split is not None
        and b.split is not None
        and not all(A.lshape_map[:, A.split] == b.lshape_map[:, 0])
    ):
        raise RuntimeError("Local arrays of A and b have different sizes.")

    if A.split is None:
        x = torch.linalg.solve_triangular(A.larray, b.larray, upper=True)
        return factories.array(x, dtype=b.dtype, device=b.device, comm=b.comm)

    nprocs = A.comm.Get_size()
    A_lshapes_cum = torch.hstack(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(A.lshape_map[:, A.split], 0)]
    )
    btilde_loc = b.larray.clone()
    x = factories.zeros_like(b, comm=b.comm)

    if A.split == 1:
        for i in range(nprocs - 1, -1, -1):
            res = torch.zeros(
                (A_lshapes_cum[i], b.shape[1]),
                dtype=b.dtype.torch_type(),
                device=b.device.torch_device,
            )
            if A.comm.rank == i:
                x.larray = torch.linalg.solve_triangular(
                    A.larray[A_lshapes_cum[i] : A_lshapes_cum[i + 1], :], btilde_loc, upper=True
                )
                res = A.larray[: A_lshapes_cum[i], :] @ x.larray
            if i > 0:
                req = A.comm.Ibcast(res, root=i)
                req.Wait()
            if A.comm.rank < i:
                j = A.comm.rank
                btilde_loc -= res[A_lshapes_cum[j] : A_lshapes_cum[j + 1], :]

    # if A.split == 1:
    #     for i in range(nprocs-1,-1,-1):
    #         count = b.lshape[0]*b.lshape[1]
    #         displ = (A_lshapes_cum*b.shape[1]).numpy()[:-1]
    #         res_send = None
    #         res_recv = torch.zeros(b.lshape[0]*b.lshape[1], dtype=b.dtype.torch_type(), device=b.device.torch_device)
    #         if A.comm.rank == i:
    #             x.larray = torch.linalg.solve_triangular(A.larray[A_lshapes_cum[i]:A_lshapes_cum[i+1],:],btilde_loc,upper=True)
    #             res_send = (A.larray @ x.larray).flatten()
    #         if i > 0:
    #             A.comm.handle.Scatterv([res_send, count, displ, MPI.DOUBLE], res_recv, root=i)
    #         if A.comm.rank < i:
    #             j = A.comm.rank
    #             btilde_loc -= res_recv.reshape(b.lshape)

    else:
        for i in range(nprocs - 1, -1, -1):
            x_from_i = torch.zeros(
                (x.lshape_map[i, 0], x.lshape_map[i, 1]),
                dtype=b.dtype.torch_type(),
                device=b.device.torch_device,
            )
            if A.comm.rank == i:
                x.larray = torch.linalg.solve_triangular(
                    A.larray[:, A_lshapes_cum[i] : A_lshapes_cum[i + 1]], btilde_loc, upper=True
                )
                x_from_i = x.larray
            if i > 0:
                req = A.comm.Ibcast(x_from_i, root=i)
                req.Wait()
            if A.comm.rank < i:
                j = A.comm.rank
                btilde_loc -= A.larray[:, A_lshapes_cum[i] : A_lshapes_cum[i + 1]] @ x_from_i

    return x


def myrandn(
    shape: Union[int, Sequence[int]],
    dtype: Type[types.datatype] = types.float32,
    split: Optional[int] = None,
    device: Optional[Device] = None,
    comm: Optional[Communication] = None,
    order: str = "C",
) -> DNDarray:
    """
    Workaround for a bug in heats randn
    """
    randint = torch.randint(0, 10000, (1,))
    torch.manual_seed(MPI.COMM_WORLD.rank + randint)

    return factories.__factory(shape, dtype, split, torch.randn, device, comm, order)
