import heat as ht
import torch
import numpy as np
import matplotlib.pyplot as plt


ht.random.seed(0)  # to get reproducibly results with the random matrices

compute_errors_to_reference = True  # if you want to see the "true" errors (possibly expensive!)
generate_plots = True  # generate some informative plots

"""
===============================================================================
CHOICE OF THE DATA
===============================================================================
"""

# matrix dimensions
m = 1000
n = 1000


# FIRST TEST DATA: own example with moderatly decaying singular values
x = ht.arange(m, dtype=ht.float64) / m
y = ht.arange(n, dtype=ht.float64) / n
yy, xx = ht.meshgrid(y, x)
A = 1.0 / (0.03 + 100 * (xx - yy) ** 2)
A.resplit_(1)


# # SECOND TEST DATA: random matrix with prescribed rank rk and eigenvalues distributed acc. to exponential distribution
# rk = 25
# A = ht.utils.data.matrixgallery.random_known_rank(m, n, rk, split=1, dtype=ht.float64)[0]


"""
===============================================================================
EXPERIMENT SETUP
===============================================================================
"""


# used for naming the plots, change this to avoid overwriting the plots
experiment_id = 1


# Parameters
maxrank = 20  # rank to which all local SVDs are truncated
# (= final truncation rank of hSVD)

maxmergedim = 70  # max. number of columns that are merged at a single process
# caveats: to small values may result in an infinite loop,
# to large values may generate memory issues
# recommendation: choose maxmergedim in the magnitude of A.lshape[1]

reltol = None  # prescribes an upper bound for the rel. "reconstruction error"
#    ||A - U Sigma V^T||_F / ||A||_F
# reltol = None is possible (if you just want to prescribe a truncation rank)
# caveats:
# * if you set a too small value for maxrank, the bound may
#   be violated at the end...
# * if you set a too small tolerance, hSVD may be inefficient or might
#   even cause memory issues

safetyshift = 5  # the truncation rank of the local SVDs inferred from
# maxrank or reltol or both is increased by safetyshift.
# It turned out that such a shift helps to improve the accuracy,
# in particular w.r.t. V

no_of_merges = 16  # at a single process at most no_of_merges SVD are merged

# HINTS:
# some of the above parameters may interact: the truncation rank of the local SVDs, e.g.,
# is determined by both maxrank and reltol. If you want to use maxrank only you need to choose
# reltol = None; if you want to use reltol only, set maxrank = matrixdimension for instance.
# maxmergedim and no_of_merges behave similarly: if you just want to prescribe the number of
# local SVDs merged at some process, use no_of_merges and set maxmergedim = matrixdimension;
# if you do not want to use no_of_merges set it to the number of MPI-processes used.


# less interesting parameters:
full = True  # means that we also compute V
# caveat: this does not mean that U, V are square matrices!
silent = False  # hSVD prints some infos on the "mergin tree"


"""
===============================================================================
COMPUTATIONS AND EVALUATION (do not modify)
===============================================================================
"""

# compute the hSVD
U, sigma, V, rel_err_est = ht.linalg.hsvd(
    A,
    maxrank=maxrank,
    maxmergedim=maxmergedim,
    reltol=reltol,
    safetyshift=safetyshift,
    no_of_merges=no_of_merges,
    full=full,
    silent=silent,
)

# check the "reconstruction error", the "orthogonality errors" and the error estimates
# this can be done in HeAT
rel_err_est = rel_err_est.numpy()[0]
A_rec = U @ ht.diag(sigma) @ V.T
true_rel_err = (ht.norm(A_rec - A) / ht.norm(A)).numpy()

hsvd_rk = sigma.shape[0]
U_orth_err = (
    ht.norm(U.T @ U - ht.eye(hsvd_rk, dtype=U.dtype, split=U.T.split, device=U.device))
    / hsvd_rk**0.5
).numpy()
V_orth_err = (
    ht.norm(V.T @ V - ht.eye(hsvd_rk, dtype=V.dtype, split=V.T.split, device=V.device))
    / hsvd_rk**0.5
).numpy()

if A.comm.rank == 0:
    print("------------------------------------------------------------------------------------")
    print("   hSVD (final rank: %d)" % hsvd_rk)
    print("------------------------------------------------------------------------------------")
    print("   Rel. reconstruction error: \t\t %2.2e" % true_rel_err)
    print(
        "   Err. est. (est./true err.): \t %2.2e (%2.4e)"
        % (rel_err_est, rel_err_est / true_rel_err)
    )
    print("   Rel. orthogonality error U: \t %2.2e" % U_orth_err)
    print("   Rel. orthogonality error V: \t %2.2e" % V_orth_err)
    print("------------------------------------------------------------------------------------")


# compare the hSVD results to the reference solution (full SVD) in numpy (may take some while!)
# caveat: output of the third SVD-factor in numpy is V.T (instead of V as in our implementation)
if compute_errors_to_reference:
    A_np = A.numpy()
    U_ref_np, sigma_ref_np, V_ref_np = np.linalg.svd(A_np)
    V_ref_np = V_ref_np.T

    U_np = U.numpy()
    V_np = V.numpy()
    sigma_np = sigma.numpy()
    V_errs = []
    U_errs = []
    # this is necessary, because in an SVD it is possible for each k to switch the sign of the kth
    # column of U and V
    for k in range(hsvd_rk):
        if np.linalg.norm(U_np[:, k] - U_ref_np[:, k]) > np.linalg.norm(U_ref_np[:, k]) / 2:
            U_ref_np[:, k] *= -1
            V_ref_np[:, k] *= -1
        U_errs.append(np.linalg.norm(U_np[:, k] - U_ref_np[:, k]))
        V_errs.append(np.linalg.norm(V_np[:, k] - V_ref_np[:, k]))

    U_rel_err = np.linalg.norm(U_np[:, :hsvd_rk] - U_ref_np[:, :hsvd_rk]) / np.linalg.norm(
        U_ref_np[:, :hsvd_rk]
    )
    V_rel_err = np.linalg.norm(V_np[:, :hsvd_rk] - V_ref_np[:, :hsvd_rk]) / np.linalg.norm(
        V_ref_np[:, :hsvd_rk]
    )
    sigma_rel_err = np.linalg.norm(sigma_np - sigma_ref_np[:hsvd_rk]) / np.linalg.norm(
        sigma_ref_np[:hsvd_rk]
    )

    if A.comm.rank == 0:
        print("   Errors w.r.t. the true SVD")
        print(
            "------------------------------------------------------------------------------------"
        )
        print("   Rel. error U: \t\t %2.2e" % U_rel_err)
        print("   Rel. error V: \t\t %2.2e" % V_rel_err)
        print("   Rel. error sigma: \t %2.2e" % sigma_rel_err)
        print(
            "------------------------------------------------------------------------------------"
        )

        if generate_plots:
            plt.figure()
            plt.title("Errors in the columns of U and V")
            plt.plot(U_errs, label="column errors U")
            plt.plot(V_errs, label="column errors V")
            plt.xlabel("no. of the column")
            plt.ylabel("error")
            plt.legend()
            plt.savefig("hsvd-column-errs-%d.jpg" % experiment_id)

            dspl_more = int(
                hsvd_rk / 3
            )  # display slightly more true sing. values than computed in hSVD

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            plt.title("True singular values vs those of hSVD")
            ax1.semilogy(
                range(len(sigma_ref_np[: hsvd_rk + dspl_more])),
                sigma_ref_np[: hsvd_rk + dspl_more],
                label="true singular values",
            )
            ax1.semilogy(range(len(sigma_np)), sigma_np, label="hSVD singular values")
            ax2.semilogy(
                range(len(sigma_np)),
                abs(sigma_np - sigma_ref_np[:hsvd_rk]) / sigma_ref_np[:hsvd_rk],
                ":",
                color="green",
                label="rel. errors (each)",
            )
            ax1.set_xlabel("no. of the singular value")
            ax1.set_ylabel("singular values")
            ax2.set_ylabel("rel. errors")
            ax1.legend()
            ax2.legend()
            plt.savefig("hsvd-singvals-errs-%d.jpg" % experiment_id)

            print(
                "See hsvd-column-errs-%d.jpg and hsvd-singvals-errs-%d.jpg for further infos."
                % (experiment_id, experiment_id)
            )
