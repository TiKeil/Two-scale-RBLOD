# ~~~
# This file is part of the paper:
#
#           " An Online Efficient Two-Scale Reduced Basis Approach
#                for the Localized Orthogonal Decomposition "
#
#   https://github.com/TiKeil/Two-scale-RBLOD.git
#
# Copyright 2019-2021 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Stephan Rave 
#   Tim Keil     
# ~~~

import numpy as np
import dill
import scipy
from scripts.problems import _construct_aFine_from_mu
from pytimings.timer import global_timings, NoTimerError


def clear_rows_and_add_unit(matrix, mask, cols=False):
    mat = matrix.tolil()
    mat[mask, :] *= 0
    if cols:
        mat[:, mask] *= 0
    mat[mask, mask] = np.ones(mat[mask, mask].shape[0])
    mat = mat.tocsr()
    return mat


def compute_coercivity(AFine, MFine, freefine=None, gridlod=False):
    if gridlod:
        return compute_coercivity_for_gridlod(AFine, MFine, freefine)
    else:
        return compute_coercivity_for_pymor(AFine, MFine)


def compute_coercivity_for_gridlod(AFine, MFine, freefine):
    # DIRICHLET TREATMENT IN A PYMOR WAY !
    # zerofine = np.array([i for i in range(AFine.shape[0]) if i not in freefine])
    # local_DoF_indices_mask = np.zeros(AFine.shape[0])
    # local_DoF_indices_mask[list(zerofine)] = 1
    # DoF_indices_mask = np.array(local_DoF_indices_mask, dtype=bool)
    # A = AFine.copy()
    # K = MFine.copy()
    # A = clear_rows_and_add_unit(A, DoF_indices_mask)
    # K = clear_rows_and_add_unit(K, DoF_indices_mask, cols=True)

    # source: A Tutorial on RB-Methods
    # see documentation of shift invert mode for smallest eigenvalue
    # --> THIS IS THE CORRECT WAY !
    A = AFine[freefine][:, freefine]
    K = MFine[freefine][:, freefine]
    # ATTENTION sigma in scipy.sparse.linalg.eigsh has a high chance of failure
    return scipy.sparse.linalg.eigsh(A, k=1, M=K, sigma=0, which="SM", return_eigenvectors=False)


def compute_coercivity_for_pymor(AFine, MFine):
    A = AFine
    K = MFine
    # source: A Tutorial on RB-Methods
    # see documentation of shift invert mode for smallest eigenvalue
    # print('WARNING: THIS MIGHT BE WRONG BECAUSE OF THE DIRICHLET DOF HANDLING IN PYMOR ! ')
    # return scipy.sparse.linalg.eigsh(A, k=1, M=K, sigma=0, which='LM', return_eigenvectors=False)
    return scipy.sparse.linalg.eigsh(A, M=K, return_eigenvectors=False)


def compute_eigvals(A, B):
    print("WARNING: THIS MIGHT BE VERY EXPENSIVE")
    return scipy.sparse.linalg.eigsh(A, M=B, which="SM", k=2, return_eigenvectors=False)


def compute_errors(AFines, H1Fine, MFine, uFEMs,
                   uLODcoarses, u_RBLODcoarses, u_TSRBLODcoarses, u_resminTSRBLODcoarses,
                   uLODfines, uRBLODfines, u_TSRBLODfines):

    assert uFEMs != []
    save_correctors = True
    if u_TSRBLODcoarses == []:
        u_TSRBLODcoarses = uFEMs
    if u_resminTSRBLODcoarses == []:
        u_resminTSRBLODcoarses = uFEMs
    if u_RBLODcoarses == []:
        u_RBLODcoarses = uFEMs
    if uLODfines == []:
        save_correctors = False
        uLODfines = uFEMs
    if uRBLODfines == []:
        uRBLODfines = uFEMs
    if u_TSRBLODfines == []:
        u_TSRBLODfines = uFEMs

    max_a_lod_RBLOD = 0
    max_a_lod_TSRBLOD = 0
    max_a_lod_resTSRBLOD = 0

    max_a_fem_RBLOD = 0
    max_a_fem_lod = 0
    max_a_fem_resTSRBLOD = 0

    max_h1_lod_RBLOD = 0
    max_h1_lod_TSRBLOD = 0
    max_h1_lod_resTSRBLOD = 0

    max_h1_fem_RBLOD = 0
    max_h1_fem_lod = 0
    max_h1_fem_resTSRBLOD = 0

    max_l2_lod_RBLOD = 0
    max_l2_lod_TSRBLOD = 0
    max_l2_lod_resTSRBLOD = 0

    max_l2_fem_RBLOD = 0
    max_l2_fem_TSRBLOD = 0
    max_l2_fem_resTSRBLOD = 0
    max_l2_lod_fem = 0

    for uFEM, uLOD, uRBLOD, uTSRBLOD, uresminTSRBLOD, uLODfine, uRBLODfine, uTSRBLODfine, AFine in zip(
        uFEMs, uLODcoarses, u_RBLODcoarses, u_TSRBLODcoarses, u_resminTSRBLODcoarses,
        uLODfines, uRBLODfines, u_TSRBLODfines, AFines):

        a_norm_fine = np.sqrt(np.dot((uFEM), AFine * (uFEM)))
        a_error_res_min_RB = np.sqrt(np.dot((uLOD - uresminTSRBLOD), AFine * (uLOD - uresminTSRBLOD)))
        a_error_new_RB = np.sqrt(np.dot((uLOD - uTSRBLOD), AFine * (uLOD - uTSRBLOD)))
        a_error_henning_RB = np.sqrt(np.dot((uLOD - uRBLOD), AFine * (uLOD - uRBLOD)))

        if save_correctors:
            a_error_fine_LOD = np.sqrt(np.dot((uFEM - uLODfine), AFine * (uFEM - uLODfine)))
            a_error_fine_RBLOD = np.sqrt(np.dot((uFEM - uRBLODfine), AFine * (uFEM - uRBLODfine)))
            a_error_fine_TSRBLOD = np.sqrt(np.dot((uFEM - uTSRBLODfine), AFine * (uFEM - uTSRBLODfine)))

        h1_norm_fine = np.sqrt(np.dot((uFEM), H1Fine * (uFEM)))
        h1_error_res_min_RB = np.sqrt(np.dot((uLOD - uresminTSRBLOD), H1Fine * (uLOD - uresminTSRBLOD)))
        h1_error_new_RB = np.sqrt(np.dot((uLOD - uTSRBLOD), H1Fine * (uLOD - uTSRBLOD)))
        h1_error_henning_RB = np.sqrt(np.dot((uLOD - uRBLOD), H1Fine * (uLOD - uRBLOD)))

        if save_correctors:
            h1_error_fine_LOD = np.sqrt(np.dot((uFEM - uLODfine), H1Fine * (uFEM - uLODfine)))
            h1_error_fine_RBLOD = np.sqrt(np.dot((uFEM - uRBLODfine), H1Fine * (uFEM - uRBLODfine)))
            h1_error_fine_TSRBLOD = np.sqrt(np.dot((uFEM - uTSRBLODfine), H1Fine * (uFEM - uTSRBLODfine)))

        l2_norm_fine = np.sqrt(np.dot((uFEM), MFine * (uFEM)))
        l2_error_res_min_RB = np.sqrt(np.dot((uLOD - uresminTSRBLOD), MFine * (uLOD - uresminTSRBLOD)))
        l2_error_new_RB = np.sqrt(np.dot((uLOD - uTSRBLOD), MFine * (uLOD - uTSRBLOD)))
        l2_error_henning_RB = np.sqrt(np.dot((uLOD - uRBLOD), MFine * (uLOD - uRBLOD)))

        l2_error_new_RB_fem = np.sqrt(np.dot((uFEM - uTSRBLOD), MFine * (uFEM - uTSRBLOD)))
        l2_error_res_min_RB_fem = np.sqrt(np.dot((uFEM - uresminTSRBLOD), MFine * (uFEM - uresminTSRBLOD)))
        l2_error_henning_RB_fem = np.sqrt(np.dot((uFEM - uRBLOD), MFine * (uFEM - uRBLOD)))
        l2_error_coarse = np.sqrt(np.dot((uFEM - uLOD), MFine * (uFEM - uLOD)))

        max_a_lod_RBLOD = max(max_a_lod_RBLOD, a_error_henning_RB/a_norm_fine)
        max_a_lod_TSRBLOD = max(max_a_lod_TSRBLOD, a_error_new_RB/a_norm_fine)
        max_a_lod_resTSRBLOD = max(max_a_lod_resTSRBLOD, a_error_res_min_RB/a_norm_fine)

        max_a_fem_RBLOD = max(max_a_fem_RBLOD, a_error_fine_RBLOD/a_norm_fine)
        max_a_fem_lod = max(max_a_fem_lod, a_error_fine_LOD/a_norm_fine)
        max_a_fem_resTSRBLOD = max(max_a_fem_resTSRBLOD, a_error_fine_TSRBLOD/a_norm_fine)

        max_h1_lod_RBLOD = max(max_h1_lod_RBLOD, h1_error_henning_RB/h1_norm_fine)
        max_h1_lod_TSRBLOD = max(max_h1_lod_TSRBLOD, h1_error_new_RB/h1_norm_fine)
        max_h1_lod_resTSRBLOD = max(max_h1_lod_resTSRBLOD, h1_error_res_min_RB/h1_norm_fine)

        max_h1_fem_RBLOD = max(max_h1_fem_RBLOD, h1_error_fine_RBLOD/h1_norm_fine)
        max_h1_fem_lod = max(max_h1_fem_lod, h1_error_fine_LOD/h1_norm_fine)
        max_h1_fem_resTSRBLOD = max(max_h1_fem_resTSRBLOD, h1_error_fine_TSRBLOD/h1_norm_fine)

        max_l2_lod_RBLOD = max(max_l2_lod_RBLOD, l2_error_henning_RB/l2_norm_fine)
        max_l2_lod_TSRBLOD = max(max_l2_lod_TSRBLOD, l2_error_new_RB/l2_norm_fine)
        max_l2_lod_resTSRBLOD = max(max_l2_lod_resTSRBLOD, l2_error_res_min_RB/l2_norm_fine)

        max_l2_fem_RBLOD = max(max_l2_fem_RBLOD, l2_error_henning_RB_fem/l2_norm_fine)
        max_l2_fem_TSRBLOD = max(max_l2_fem_TSRBLOD, l2_error_new_RB_fem/l2_norm_fine)
        max_l2_fem_resTSRBLOD = max(max_l2_fem_resTSRBLOD, l2_error_res_min_RB_fem/l2_norm_fine)
        max_l2_lod_fem = max(max_l2_lod_fem, l2_error_coarse/l2_norm_fine)

    print("\n ***************************************************** ")

    print("\nMAX RELATIVE ENERGY ERRORS \n")
    print(f"relative energy error of henning RBLOD vs coarse LOD:    {max_a_lod_RBLOD:.12f} ")
    print(f"relative energy error of res TSRBLOD vs coarse LOD:      {max_a_lod_resTSRBLOD:.12f} ")
    print(f"relative energy error of TSRBLOD vs coarse LOD:          {max_a_lod_TSRBLOD:.12f} ")
    if save_correctors is not None:
        print(f"\nrelative energy error of fine LOD vs FEM:                {max_a_fem_lod:.12f} ")
        print(f"relative energy error of fine RBLOD vs FEM:              {max_a_fem_RBLOD:.12f} ")
        print(f"relative energy error of fine TSRBLOD vs FEM:            {max_a_fem_resTSRBLOD:.12f} ")

    print("\n ***************************************************** ")

    print("\nMAX RELATIVE H1 ERRORS \n")
    print(f"relative h1 error of henning RBLOD vs coarse LOD:    {max_h1_lod_RBLOD:.12f} ")
    print(f"relative h1 error of res TSRBLOD vs coarse LOD:      {max_h1_lod_resTSRBLOD:.12f} ")
    print(f"relative h1 error of TSRBLOD vs coarse LOD:          {max_h1_lod_TSRBLOD:.12f} ")
    if save_correctors is not None:
        print(f"\nrelative h1 error of fine LOD vs FEM:                {max_h1_fem_lod:.12f} ")
        print(f"relative h1 error of fine RBLOD vs FEM:              {max_h1_fem_RBLOD:.12f} ")
        print(f"relative h1 error of fine TSRBLOD vs FEM:            {max_h1_fem_resTSRBLOD:.12f} ")

    print("\n ***************************************************** ")

    print("\nMAX RELATIVE L2 ERRORS \n")
    print(f"relative l2 error of henning RBLOD vs coarse LOD:    {max_l2_lod_RBLOD:.12f} ")
    print(f"relative l2 error of res TSRBLOD vs coarse LOD:      {max_l2_lod_resTSRBLOD:.12f} ")
    print(f"relative l2 error of TSRBLOD vs coarse LOD:          {max_l2_lod_TSRBLOD:.12f} ")
    print(f"\nrelative l2 error of henning RBLOD vs FEM:           {max_l2_fem_RBLOD:.12f} ")
    print(f"relative l2 error of res TSRBLOD vs FEM:             {max_l2_fem_resTSRBLOD:.12f} ")
    print(f"relative l2 error of TSRBLOD vs FEM:                 {max_l2_fem_TSRBLOD:.12f} ")
    print(f"relative l2 error of coarse LOD vs FEM:              {max_l2_lod_fem:.12f} ")


def compute_coarse_errors(H1Coarse, MCoarse, uLodCoarses, u_RBLODs, u_TSRBLODs, u_resTSRBLODs):
    assert uLodCoarses != []
    if u_RBLODs == []:
        u_RBLODs = uLodCoarses
    if u_TSRBLODs == []:
        u_TSRBLODs = uLodCoarses
    if u_resTSRBLODs == []:
        u_resTSRBLODs = uLodCoarses

    max_h1_lod_RBLOD = 0
    max_h1_lod_TSRBLOD = 0
    max_h1_lod_resTSRBLOD = 0

    max_l2_lod_RBLOD = 0
    max_l2_lod_TSRBLOD = 0
    max_l2_lod_resTSRBLOD = 0

    for uLodCoarse, u_RBLOD, u_TSRBLOD, u_resTSRBLOD in zip(uLodCoarses, u_RBLODs, u_TSRBLODs, u_resTSRBLODs):
        h1_norm_coarse = np.sqrt(np.dot((uLodCoarse), H1Coarse * (uLodCoarse)))
        h1_error_resTSRBLOD = np.sqrt(np.dot((uLodCoarse - u_resTSRBLOD), H1Coarse * (uLodCoarse - u_resTSRBLOD)))
        h1_error_TSRBLOD = np.sqrt(np.dot((uLodCoarse - u_TSRBLOD), H1Coarse * (uLodCoarse - u_TSRBLOD)))
        h1_error_RBLOD = np.sqrt(np.dot((uLodCoarse - u_RBLOD), H1Coarse * (uLodCoarse - u_RBLOD)))

        l2_norm_coarse = np.sqrt(np.dot((uLodCoarse), MCoarse * (uLodCoarse)))
        l2_error_resTSRBLOD = np.sqrt(np.dot((uLodCoarse - u_resTSRBLOD), MCoarse * (uLodCoarse - u_resTSRBLOD)))
        l2_error_TSRBLOD = np.sqrt(np.dot((uLodCoarse - u_TSRBLOD), MCoarse * (uLodCoarse - u_TSRBLOD)))
        l2_error_RBLOD = np.sqrt(np.dot((uLodCoarse - u_RBLOD), MCoarse * (uLodCoarse - u_RBLOD)))

        max_h1_lod_RBLOD = max(max_h1_lod_RBLOD, h1_error_RBLOD/h1_norm_coarse)
        max_h1_lod_TSRBLOD = max(max_h1_lod_TSRBLOD, h1_error_TSRBLOD/h1_norm_coarse)
        max_h1_lod_resTSRBLOD = max(max_h1_lod_resTSRBLOD, h1_error_resTSRBLOD/h1_norm_coarse)

        max_l2_lod_RBLOD = max(max_l2_lod_RBLOD, l2_error_RBLOD/l2_norm_coarse)
        max_l2_lod_TSRBLOD = max(max_l2_lod_TSRBLOD, l2_error_TSRBLOD/l2_norm_coarse)
        max_l2_lod_resTSRBLOD = max(max_l2_lod_resTSRBLOD, l2_error_resTSRBLOD/l2_norm_coarse)

    print("\n ***************************************************** ")

    print("\nMAX RELATIVE H1 ERRORS \n")
    print(f"relative h1 error of henning RBLOD vs coarse LOD:    {max_h1_lod_RBLOD:.12f} ")
    print(f"relative h1 error of res TSRBLOD vs coarse LOD:      {max_h1_lod_resTSRBLOD:.12f} ")
    print(f"relative h1 error of TSRBLOD vs coarse LOD:          {max_h1_lod_TSRBLOD:.12f} ")

    print("\n ***************************************************** ")

    print("\nMAX RELATIVE L2 ERRORS \n")
    print(f"relative l2 error of henning RBLOD vs coarse LOD:    {max_l2_lod_RBLOD:.12f} ")
    print(f"relative l2 error of res TSRBLOD vs coarse LOD:      {max_l2_lod_resTSRBLOD:.12f} ")
    print(f"relative l2 error of TSRBLOD vs coarse LOD:          {max_l2_lod_TSRBLOD:.12f} ")


def estimator_printout(max_full, max_fine, max_coarse):
    print("\n ***************************************************** ")

    print("\nMAX Estimator values \n")
    print(f"Maximal certified estimator value:   {max_full[0]:.12f}")
    print(f"Maximal fine estimator value:        {max_fine[0]:.12f}")
    print(f"Maximal coarse estimator value:      {max_coarse[0]:.12f}")


def rom_size_printout(rom_sizeT_, extension_failedT_henning, rom_sizeT, extension_failedT, rom_two_scale,
                      extension_failed, verbose=False):
    print("\nSIZES OF THE ROMS ")
    if verbose:
        print("size of henning stage 1 roms:        {}".format(rom_sizeT_))
    print("total size of henning stage 1:       {}".format(np.sum(rom_sizeT_)), end="")
    print(" (extension failed in at least one training)") if np.sum(extension_failedT_henning) else print(
        " (all succeeded)"
    )
    print("average size of henning stage 1:     {}".format(np.sum(rom_sizeT_)/len(rom_sizeT_)/4))
    if verbose:
        print("\nsize of TSRBLOD stage 1 roms:        {}".format(rom_sizeT))
    print("total size of TSRBLOD stage 1:       {}".format(np.sum(rom_sizeT)), end="")
    print(" (extension failed in at least one training)") if np.sum(extension_failedT) else print(" (all succeeded)")
    print("average size of TSRBLOD stage 1:     {}".format(np.sum(rom_sizeT)/len(rom_sizeT)))
    if rom_two_scale is None:
        print("size of new two scale rom:           0", end="")
    else:
        print(
            "size of new two scale rom:           {}".format(rom_two_scale.operator.source.dim),
            end="",
        )
    print(" (extension failed)") if extension_failed else print(" (succeeded)")


def times_printout(NWorldCoarse, timeT_=[np.inf], timeT=[np.inf]):
    get = global_timings.walltime
    print("\nTIMINGS OFFLINE ")
    try:
        print(f"total time of parallel henning stage 1:     {get('offline henning stage 1'):.3f} seconds")
    except NoTimerError:
        pass
    print(
        f"average time of single henning stage 1:     "
        f"{np.sum(timeT_)/np.prod(NWorldCoarse):.3f} seconds"
    )
    print(f"total time of parallel two-scale stage 1:   {get('offline stage 1'):.3f} seconds")
    print(
        f"average time of single TSRBLOD stage 1:     "
        f"{np.sum(timeT)/np.prod(NWorldCoarse):.3f} seconds"
    )
    print(f"total time of certified two-scale stage 2:  {get('offline stage 2'):.3f} seconds")
    print(f"total time of TSRBLOD                       {get('offline stage 2')+get('offline stage 1'):.3f} seconds")

    print("\nCOMPARE ONLINE TIMINGS ")
    full_LOD_online_time = get('online step 1 LOD') + get('online step 2') + get('online step 3')
    full_henning_online_time = get('online step 1 RBLOD') + get('online step 2') + get('online step 3')
    speed_up_RBLOD = full_LOD_online_time/full_henning_online_time
    speed_up_TSRBLOD = full_LOD_online_time/get('online TSRBLOD')
    speed_up_resTSRBLOD = full_LOD_online_time / get('online resmin TSRBLOD')
    print(f"time for FEM:                       {get('online FEM'):.5f} seconds")
    print(f"time for PG-LOD                     {full_LOD_online_time:.5f} seconds")
    print(f"time for henning RBLOD              {full_henning_online_time:.5f} seconds --> speed-up: "
          f"{speed_up_RBLOD:.3f}")
    print(f"time for TSRBLOD                    {get('online TSRBLOD'):.5f} seconds --> speed-up: "
          f"{speed_up_TSRBLOD:.3f}")
    print(f"time for residual min TSRBLOD       {get('online resmin TSRBLOD'):.5f} seconds --> speed-up: "
          f"{speed_up_resTSRBLOD:.3f}")

    print("\nAdditional information on henning RB: ")
    print(f"Step 1: Assembly of all K_i parts:  {get('online step 1 RBLOD'):.5f} seconds")
    print(f"Step 2: Assembly of global Kms:     {get('online step 2'):.5f} seconds")
    print(f"Step 3: Solve the coarse system:    {get('online step 3'):.5f} seconds")
    print(f"In total:                           {full_henning_online_time:.5f} seconds")


def extended_time_printout(timeT, timeT_):
    print('all local times for henning:')
    print('  ', end='', flush=True)
    for t_ in timeT_:
        print(f'{t_:.3f}s ', end='', flush=True)
    print()
    print('all local times for two scale:')
    print('  ', end='', flush=True)
    for t_ in timeT:
        print(f'{t_:.3f}s ', end='', flush=True)
    print()


def verbose_stage_printout(verbose, max_error_mus, max_errors, print_params=False, N=0):
    if verbose:
        if N:
            print("PARAMETERS FOR REDUCTION: ")
            for i in range(int(N * N)):
                print("\n     T {}:     ".format(i), end="")
                if print_params:
                    for mu_ in max_error_mus[i]:
                        print("{}, \n                ".format(mu_.to_numpy()), end="")
                print("\n with errors: ", end="")
                for mu_ in max_errors[i]:
                    print("{:.12f}, ".format(mu_), end="")
            print("\nextensions: {}".format(len(max_errors)))
        else:
            print("PARAMETERS FOR REDUCTION: \n")
            if print_params:
                print("           mus: ", end="")
                for mu_ in max_error_mus:
                    print("{}, \n                ".format(mu_.to_numpy()), end="")
            print("\n   with errors: ", end="")
            for mu_ in max_errors:
                print("{:.12f}, \n                ".format(mu_), end="")
            print("\n    extensions: {}".format(len(max_errors)))


def storeData(dict, filename="pickle_data.pkl"):
    # Its important to use binary mode
    dbfile = open(filename, "ab")
    # source, destination
    dill.dump(dict, dbfile)
    dbfile.close()


def loadData(filename="pickle_data.pkl"):
    # for reading also binary mode is important
    dbfile = open(filename, "rb")
    dict = dill.load(dbfile)
    dbfile.close()
    return dict


def compute_constrast(aFines, aFineCoefficients, training_set):
    max_contrast = 1
    min_alpha = 10000
    for mu in training_set:
        a = _construct_aFine_from_mu(aFines, aFineCoefficients, mu)
        if a.ndim == 3:
            a = np.linalg.norm(a, axis=(1, 2), ord=2)
        max_contrast = max(max_contrast, np.max(a) / np.min(a))
        min_alpha = min(min_alpha, np.min(a))
    return max_contrast, min_alpha
