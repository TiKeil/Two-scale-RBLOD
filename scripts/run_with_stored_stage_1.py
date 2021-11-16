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

import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.sparse as sparse

### pytimings
from pytimings.timer import global_timings as timings, scoped_timing
import time

### gridlod
from gridlod import pglod, util, fem, femsolver
from gridlod.world import World, Patch

### pymor
from pymor.algorithms.greedy import rb_greedy
from pymor.basic import set_log_levels, set_defaults
from pymor.models.basic import StationaryModel
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import IdentityOperator
from pymor.parallel.mpi import MPIPool
from pymor.parallel.dummy import DummyPool
from pymor.parameters.base import ParameterSpace
from pymor.tools import mpi
from pymor.vectorarrays.numpy import NumpyVectorSpace

### RBLOD
from rblod.parameterized_stage_1 import correctors_from_TS_RBLOD_approach
from rblod.separated_stage_1 import henning_RBLOD_approach
from rblod.optimized_rom import OptimizedTwoScaleNumpyModel
from rblod.two_scale_model import Two_Scale_Problem
from rblod.two_scale_reductor import CoerciveRBReductorForTwoScale, TrueDiagonalBlockOperator

### from scripts
from scripts.tools import loadData
from scripts.problems import model_problem_1, layer_problem_1, _construct_aFine_from_mu
from scripts.tools import verbose_stage_printout, times_printout, rom_size_printout
from scripts.tools import compute_constrast, compute_coarse_errors, compute_errors

"""
########################################################################################################
                                            CONFIG
########################################################################################################
"""
use_mpi = False
verbose = False
save_correctors = True
use_fine_mesh = True
certified_estimator_study = False
henning = True
two_scale = True
train_for = 'full'

"""
call this skript via:
    > python run_with_stored_stage_1.py 2 4 0.001 0.001 2 0 [--mpi] [--v] [--sc] [--oc] [--ces] [--sh] [--sts]
where:
    arg_1 : n_h
    arg_2 : n_H 
    arg_3 : epsilon_1
    arg_4 : epsilon_2
    arg_5 : size of verification set
    arg_6 : problem class :  --> 0: model problem 1
                             --> 1: model problem 2
    arg_7 : enrich stage 2 until a specific number. This makes epsilon_2 obsolete.
    
    --mpi : use MPI pool (use with "mpirun -n 4" in front)
    --v   : generate verbose output
    --sc  : do not save fine correctors at all
    --oc  : only use computations that work on the coarse scale. No FEM, no full visualization, no fine errors.
    --ces : prepare for the estimator study of the certified error estimator in the TSRBLOD
    --sh  : do not construct and compare Henning RBLOD method
    --sts : do not construct and compare TSRBLOD method
    --uce : use coarse error for two-scale model (only for estimator study purpose)
"""

print()
assert isinstance(int(sys.argv[1]), int)                # --> n_h
assert isinstance(int(sys.argv[2]), int)                # --> n_H
assert isinstance(float(sys.argv[3]), float)            # --> epsilon_1
assert isinstance(float(sys.argv[4]), float)            # --> epsilon_2
assert isinstance(int(sys.argv[5]), int)                # --> size of verification set
assert int(sys.argv[6]) == 0 or int(sys.argv[6]) == 1   # --> 0: model problem 1     --> 1: model problem 2
try:
    int(sys.argv[7])
    extend_until = int(sys.argv[7])
except:
    extend_until = None

if "--mpi" in sys.argv:
    print("Using MPI parallel run, make sure to use mpirun.")
    use_mpi = True
    pool = MPIPool()
else:
    pool = DummyPool()
if "--sld" in sys.argv:
    print("store local element data (and do not communicate it)")
    store = True
if "--p" in sys.argv:
    print("stage 1 data will be pickled")
    pickle_data = True
if "--v" in sys.argv or "--verbose" in sys.argv:
    print("using verbose printout")
    verbose = True
if "--sc" in sys.argv or "--skip-correctors" in sys.argv:
    print("always remove correctors from storage")
    save_correctors = False
if "--oc" in sys.argv or "--only-coarse" in sys.argv:
    print("not using any full fine data")
    use_fine_mesh = False
if "--ces" in sys.argv or "--certified-estimator-study" in sys.argv:
    print("also show the behavior of the estimator")
    certified_estimator_study = True
if "--sh" in sys.argv or "--skip-henning" in sys.argv:
    print("skip henning method entirely")
    henning = False
if "--sts" in sys.argv or "--skip-two-scale" in sys.argv:
    print("skip two scale method entirely")
    two_scale = False
if "--uce" in sys.argv or "--use-coarse-estimator" in sys.argv:
    print("use the coarse estimator instead of the certified for the two-scale model")
    train_for = 'coarse'

if use_fine_mesh is False:
    assert save_correctors is False, "You can not store correctors in this case"

def prepare():
    if verbose is True:
        set_log_levels({"pymor": "INFO"})
    else:
        set_log_levels({"pymor": "WARN"})
    set_defaults({"pymor.algorithms.gram_schmidt.gram_schmidt.rtol": 1e-8})
    set_defaults({"pymor.algorithms.gram_schmidt.gram_schmidt.check": False})
    np.warnings.filterwarnings("ignore")  # silence numpys warnings

pool.apply(prepare)
path = ""

"""
########################################################################################################
                                PROBLEM SETUP AND LOD VARIABLES
########################################################################################################
"""
if int(sys.argv[6]) == 0:
    experiment = "henning"
elif int(sys.argv[6]) == 1:
    experiment = "layer"

# parameters for the grid size
N = int(sys.argv[1])
n = int(sys.argv[2])
k = int(np.ceil(np.abs(np.log(np.sqrt(2) * 1./N)))) # Localization parameter of patches

atol_patch = float(sys.argv[3])                                 # epsilon_1
atol_two_scale = 1e-16 if extend_until else float(sys.argv[4])  # epsilon_2
verification_size = int(sys.argv[5])                            # size of verification set

NFine = np.array([n, n])                            # n x n fine grid elements
NpFine = np.prod(NFine + 1)                         # Number of fine DoFs
NWorldCoarse = np.array([N, N])                     # N x N coarse grid elements
boundaryConditions = np.array([[0, 0], [0, 0]])     # zero Dirichlet boundary conditions
NCoarseElement = NFine // NWorldCoarse
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)  # gridlod specific class

print("\nSTARTING SCRIPT ...\n")
print(f"Coarse FEM mesh:            {N} x {N}")
print(f"Fine FEM mesh:              {n} x {n}")
print(f"k:                          {k}")
print(f"|log H|:                    {np.abs(np.log(np.sqrt(2) * 1./N)):.2f}")
print(f"number of fine dofs         {NpFine}")
middle_coarse_index = np.prod(NWorldCoarse)//2 + NWorldCoarse[0]//2
print(f"max fine dofs per patch:    {Patch(world, k, middle_coarse_index).len_fine}")
print(f"number of parallel kernels: {mpi.size} ")
print(f"\nGREEDY TOLERANCES: \nStage 1:   {atol_patch}\nTwo-scale: {atol_two_scale}\n")
print(f'Extend two-scale until: {extend_until}\n')

if experiment == "henning":
    param_min, param_max = 0, 5
    aFines, aFineCoefficients, f, f_fine, model_parameters, aFine_Constructor = \
        model_problem_1(NFine, world, plot=False, return_fine=use_fine_mesh)
    training_number = 50
elif experiment == "layer":
    param_min, param_max = 1, 5
    aFines, aFineCoefficients, f, f_fine, model_parameters, aFine_Constructor = \
        layer_problem_1(NFine, world, coefficients=3, plot=False, return_fine=use_fine_mesh)
    training_number = 4

# standard parameter space
parameter_space = ParameterSpace(model_parameters, [param_min, param_max])
verification_set = parameter_space.sample_randomly(verification_size)
if certified_estimator_study:
    verification_set = []

"""
########################################################################################################
                                            OFFLINE PHASE
########################################################################################################
"""

"""
Construct training sets
"""
print(f"Training set size per dimension:     {training_number}")

training_set = parameter_space.sample_uniformly(training_number)

def construct_patches(TInd, k, world):
    patch = Patch(world, k, TInd)
    return patch


def compute_contrast_for_all_patches(patch, aFineCoefficients, training_set, aFine_Constructor):
    aPatches = aFine_Constructor(patch)
    contrast, min_alpha = compute_constrast(aPatches, aFineCoefficients, training_set)
    return contrast, min_alpha


#localize coefficient beforehand
print('prepare patches ... \n')
coarse_indices = range(world.NtCoarse)
patchT = pool.map(construct_patches, list(coarse_indices), k=k, world=world)

print('computing contrast ... ')
contrasts, min_alphas = zip(*pool.map(compute_contrast_for_all_patches, patchT,
                                      aFineCoefficients=aFineCoefficients, training_set=training_set,
                                      aFine_Constructor=aFine_Constructor))
contrast, min_alpha = np.max(contrasts), np.min(min_alphas)
print(f"contrast: {contrast},   min_alpha: {min_alpha}")
coercivity_estimator = lambda mu: min_alpha

# separate gather data size for | N_H | > CPUS
if use_mpi and mpi.size < len(coarse_indices):
    split_into = len(coarse_indices) // mpi.size
    print(f'\n ... splitting processes into {split_into} (parallel) subprocesses')
    size_of_split = len(coarse_indices) // split_into
    coarse_index_list, coarse_patch_list = [], []
    for i in range(split_into):
        a, b = i * size_of_split, (i + 1) * size_of_split
        coarse_patch_list.append(patchT[a:b])
        coarse_index_list.append(coarse_indices[a:b])
else:
    coarse_patch_list, coarse_index_list = [patchT], [coarse_indices]

"""
########################################################################################################
                                STAGE 1: reducing corrector problems
                                            LOAD DATA
########################################################################################################
"""
print("\n............ BUILDING RB Models for Corrector Problems...........\n ")

"""
Offline phase for separated Henning stage 1 
"""
if henning:
    data = loadData(f"{path}pickle_data/{experiment}_pickle_data_henning_{N}_{n}_{int(np.log10(1 / atol_patch))}")
    optimized_romT_, rom_sizeT_, max_errorsT_, max_error_musT_, extension_failedT_henning, bases_, time_ = (
        data.rom, data.size, data.max_err, data.max_err_mus, data.ext, data.bases, data.total_time
    )
    timings.add_walltime("offline henning stage 1", time_)

"""
Offline phase for two scale parameterized stage 1 
"""
if two_scale:
    data = loadData(f"{path}pickle_data/{experiment}_pickle_data_{N}_{n}_{int(np.log10(1 / atol_patch))}")
    optimized_romT, rom_sizeT, max_errorsT, max_error_musT, KijT, extension_failedT, basesT, productT, h1productT, time_ \
        = ( data.rom, data.size, data.max_err, data.max_err_mus, data.K, data.ext, data.bases, data.product,
            data.h1product, data.total_time )
    timings.add_walltime("offline stage 1", time_)
"""
########################################################################################################
                                STAGE 2: reducing two-scale reduced PGLOD model
########################################################################################################
"""
print("............ BUILDING TWO SCALE RBLOD PROBLEM ...........\n")

# C = sqrt(5) * gamma_k^(-1) = sqrt(5) * C_IH * 1/sqrt(alpha)
# gamma_k^(-1) = C_IH * 1/sqrt(alpha)
C_IH = 1 # always C_IH > 1. and  C_IH /approx 1 for quadrilateral meshes
gamma_k = np.sqrt(min_alpha) * 1/C_IH
C = np.sqrt(5) * 1/gamma_k
constants = lambda mu: 1 / C   # needs to be inverted for pymor
print(f"contrast: {contrast},   min_alpha: {min_alpha}")
if two_scale:
    m_two_scale = Two_Scale_Problem(optimized_romT, KijT, f, patchT, contrast, min_alpha)


print("\n..... REDUCTOR WITH CERTIFIED ERROR ESTIMATOR ...........\n")
if two_scale:
    H1Coarse = fem.assemblePatchMatrix(NWorldCoarse, world.ALocCoarse)[m_two_scale.free][:, m_two_scale.free]
    blocks = [NumpyMatrixOperator(H1Coarse)]
    for size in rom_sizeT:
        blocks.append(IdentityOperator(NumpyVectorSpace(size)))
    two_scale_product = TrueDiagonalBlockOperator(blocks, only_first=True)
    reductor_two_scale = CoerciveRBReductorForTwoScale(
        world, optimized_romT, m_two_scale, coercivity_estimator=constants, product=two_scale_product,
        check_orthonormality=False, train_for=train_for
    )

    set_log_levels({'pymor': 'INFO'})
    greedy_data_two_scale = rb_greedy(
        m_two_scale, reductor_two_scale, training_set=training_set, max_extensions=extend_until,
        atol=atol_two_scale, extension_params={"method": "gram_schmidt"}
    )
    set_log_levels({'pymor': 'WARN'})
    time.sleep(1)

    rom_two_scale = greedy_data_two_scale["rom"]
    rom_two_scale_optimized = OptimizedTwoScaleNumpyModel(rom_two_scale)

    extension_failed = False if greedy_data_two_scale["max_errs"][-1] < atol_two_scale else True
    timings.add_walltime("offline stage 2", greedy_data_two_scale["time"])

    verbose_stage_printout(True, greedy_data_two_scale["max_err_mus"], greedy_data_two_scale["max_errs"], True, N=0)

    class ResidualMinimizingModel(StationaryModel):
        def _compute_solution(self, mu=None, **kwargs):
            return self.operator.apply_inverse(self.rhs.as_range_array(mu), mu=mu, least_squares=True)

    reduced_residual = reductor_two_scale.full_residual_reductor.reduce()
    residual_min_two_scale_rom = ResidualMinimizingModel(reduced_residual.operator, reduced_residual.rhs)
    residual_min_two_scale_rom_optimized = OptimizedTwoScaleNumpyModel(residual_min_two_scale_rom)

    print("\n ... TWO SCALE MODEL COMPLETED ! \n")

    rho_sqrt = m_two_scale.rho_sqrt
    ### No full order model needs to be stored
    del m_two_scale
    verbose_stage_printout(verbose, max_error_musT, max_errorsT, N=N)

"""
########################################################################################################
                                    ONLINE PHASE AND VERIFICATION
########################################################################################################
"""

print("Parameters used for verification:")
for mu_ver in verification_set:
    print(f"  {mu_ver.to_numpy()}")
print()

"""
#########################
 FEM reference solution 
#########################
"""
uFEMs = []
if use_fine_mesh and not certified_estimator_study:
    for mu_ver in verification_set:
        aFine = _construct_aFine_from_mu(aFines, aFineCoefficients, mu_ver)
        with scoped_timing("online FEM", print, format='.5f'):
            uFineFull, _, MFine, H1Fine, _ = femsolver.solveFine(
                world, aFine, f_fine, None, boundaryConditions, return_fine=True)
        uFEMs.append(uFineFull)
    timings.add_walltime("online FEM", timings.walltime("online FEM")/len(verification_set))
else:
    timings.add_walltime("online FEM", np.inf)
    MFine, H1Fine = None, None

"""
#########################
 Standard PG-LOD 
#########################
"""

def reference_solution(patch, boundaryConditions, save_correctors, aFineCoefficients, mu, aFine_Constructor):
    """
    classic PG-LOD patch computation on an element
    """
    from gridlod import lod, interp
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    aPatches = aFine_Constructor(patch)
    aPatch = _construct_aFine_from_mu(aPatches, aFineCoefficients, mu)
    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    if not save_correctors:
        correctorsList = None
    return np.array([csi.Kmsij, correctorsList])

print("\nComputing reference LOD ...")

basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
# right hand side can be assembled offline
free = util.interiorpIndexMap(NWorldCoarse)
MFull = fem.assemblePatchMatrix(NWorldCoarse, world.MLocCoarse)
bFull = MFull * f
bFree = bFull[free]

xFulls, uLODfines = [], []
for mu_ver in verification_set:
    print("       --> ", end="", flush=True)
    with scoped_timing("online step 1 LOD", print, format='.5f'):
        KmsijT, correctorsListT = zip(*pool.map(reference_solution, patchT, boundaryConditions=boundaryConditions,
                                                save_correctors=save_correctors, mu=mu_ver,
                                                aFine_Constructor=aFine_Constructor,
                                                aFineCoefficients=aFineCoefficients))
    with scoped_timing("online step 2"):  # step 2 in the paper
        KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
    with scoped_timing("online step 3"):  # step 3 in the paper
        KFree = KFull[free][:, free]
        xFree = sparse.linalg.spsolve(KFree, bFree)
    xFull = np.zeros(world.NpCoarse)
    xFull[free] = xFree
    xFulls.append(xFull)

    if save_correctors:
        # classic LOD
        basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
        modifiedBasis = basis - basisCorrectors
        uLODfines.append(modifiedBasis * xFull)

if not certified_estimator_study:
    timings.add_walltime("online step 1 LOD", timings.walltime("online step 1 LOD") / verification_size)
    timings.add_walltime("online step 2", timings.walltime("online step 2") / verification_size)
    timings.add_walltime("online step 3", timings.walltime("online step 3") / verification_size)

"""
#########################
 Online RBLOD
#########################
"""
xFull_henning_RBs, uRBLODfines = [], []
if henning and not certified_estimator_study:
    print("\nComputing stage 1 henning RBLOD ...")
    assert np.alltrue(rom_sizeT_), "There exists a rom with an empty basis"
    for mu_ver in verification_set:
        print("       --> ", end="", flush=True)
        def henning_RBLOD_approach_(rom):
            return henning_RBLOD_approach(rom, mu_ver)
        with scoped_timing("online step 1 RBLOD", print, format='.5f'):
            _, _ = zip(*map(henning_RBLOD_approach_, list(optimized_romT_)))
        with scoped_timing("online step 1 RBLOD parallel"):
            KmsijT_henning, correctorList_henning = zip(*pool.map(henning_RBLOD_approach, list(optimized_romT_),
                                                                  mu=mu_ver))
        print(f"                                 --> parallel computation took "
              f"{timings.walltime('online step 1 RBLOD parallel'):.5f}s")

        # identical time to the above code step 2 and step 3
        KFull_henning_RB = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT_henning)
        KFree_henning_RB = KFull_henning_RB[free][:, free]
        xFree_henning_RB = sparse.linalg.spsolve(KFree_henning_RB, bFree)
        xFull_henning_RB = np.zeros(world.NpCoarse)
        xFull_henning_RB[free] = xFree_henning_RB
        xFull_henning_RBs.append(xFull_henning_RB)

        if save_correctors:
            correctorsListT_henning = ()
            for cT, basisT in zip(correctorList_henning, bases_):
                all_correctors = []
                for c_, basis_ in zip(cT, basisT):
                    all_correctors.append(basis_.lincomb(c_).to_numpy()[0])
                correctorsListT_henning += (all_correctors,)
            basisCorrectors_henning_RB = pglod.assembleBasisCorrectors(world, patchT, correctorsListT_henning)
            modifiedBasis_henning_RB = basis - basisCorrectors_henning_RB
            uRBLODfines.append(modifiedBasis_henning_RB * xFull_henning_RB)

    timings.add_walltime("online step 1 RBLOD", timings.walltime("online step 1 RBLOD") / verification_size)

"""
#########################
 Online TSRBLOD
#########################
"""
u_TSRBLODs, u_resminTSRBLODs = [], []
if two_scale and not certified_estimator_study:
    # with standard solver
    for mu_ver in verification_set:
        with scoped_timing("online TSRBLOD"):
            U_ROM = rom_two_scale_optimized.solve(mu_ver)
            u_rom = reductor_two_scale.reconstruct_partial(U_ROM)
        u_H_two_scale = np.zeros(world.NpCoarse)
        u_H_two_scale[free] = u_rom.to_numpy()[0]
        u_TSRBLODs.append(u_H_two_scale)
    timings.add_walltime("online TSRBLOD", timings.walltime("online TSRBLOD") / verification_size)

    # with residual minimization (used in publication)
    print("\nComputing TSRBLOD ...")
    for mu_ver in verification_set:
        print("       --> ", end="", flush=True)
        with scoped_timing("online resmin TSRBLOD", print, format='.5f'):
            U_ROM = residual_min_two_scale_rom_optimized.solve(mu_ver, least_squares=True)
            u_rom = reductor_two_scale.reconstruct_partial(U_ROM)
        u_H_two_scale_res_min = np.zeros(world.NpCoarse)
        u_H_two_scale_res_min[free] = u_rom.to_numpy()[0]
        u_resminTSRBLODs.append(u_H_two_scale_res_min)
    timings.add_walltime("online resmin TSRBLOD", timings.walltime("online resmin TSRBLOD") / verification_size)

"""
#########################
 Prolongation to the fine grid (optional) 
#########################
"""
uLODcoarses, u_RBLODcoarses, u_TSRBLODcoarses, u_resminTSRBLODcoarses, u_TSRBLODfines = [], [], [], [], []
AFines = []
if use_fine_mesh and not certified_estimator_study:
    # without correctors
    basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
    if two_scale:
        for u, u_res in zip(u_TSRBLODs, u_resminTSRBLODs):
            u_TSRBLODcoarses.append(basis * u)
            u_resminTSRBLODcoarses.append(basis * u_res)
    if henning:
        for xFull_henning_RB in xFull_henning_RBs:
            u_RBLODcoarses.append(basis * xFull_henning_RB)
    for xFull in xFulls:
        uLODcoarses.append(basis * xFull)

    # with correctors
    if save_correctors:
        for mu_ver, u_H_two_scale in zip(verification_set, u_resminTSRBLODs):
            if two_scale:
                correctorsListT_TS_RBLOD = ()
                correctorsListT_ = pool.map(correctors_from_TS_RBLOD_approach, list(optimized_romT), mu=mu_ver)
                for cT, basis_ in zip(correctorsListT_, basesT):
                    all_correctors = []
                    for c_ in cT:
                        all_correctors.append(basis_.lincomb(c_).to_numpy()[0])
                    correctorsListT_TS_RBLOD += (all_correctors,)
                basisCorrectors_TS_RBLOD = pglod.assembleBasisCorrectors(world, patchT, correctorsListT_TS_RBLOD)
                modifiedBasis_TS_RBLOD = basis - basisCorrectors_TS_RBLOD
                u_TSRBLODfines.append(modifiedBasis_TS_RBLOD * u_H_two_scale)

    # construct the energy norm
    if aFine.ndim == 1:
        ALocFine = world.ALocFine
    else:
        ALocFine = world.ALocMatrixFine
    for mu_ver in verification_set:
        aFine = _construct_aFine_from_mu(aFines, aFineCoefficients, mu_ver)
        AFines.append(fem.assemblePatchMatrix(NFine, ALocFine, aFine))


"""
########################################################################################################
                                                RESULTS
########################################################################################################
"""
if not two_scale:
    assert henning
    rom_sizeT = [0 for i in rom_sizeT_]
    extension_failedT = [0 for i in rom_sizeT_]
    extension_failed = 0
    timings.add_walltime("offline stage 1", np.inf)
    timings.add_walltime("offline stage 2", np.inf)
    timings.add_walltime("online TSRBLOD", np.inf)
    timings.add_walltime("online resmin TSRBLOD", np.inf)
    rom_two_scale = None

if not henning:
    assert two_scale
    rom_sizeT_ = [0 for i in rom_sizeT]
    extension_failedT_henning = [0 for i in rom_sizeT]
    timings.add_walltime("offline henning stage 1", np.inf)
    timings.add_walltime("online step 1 RBLOD", np.inf)

"""
#########################
 Times and ROM sizes
#########################
"""
if not certified_estimator_study:
    rom_size_printout(rom_sizeT_, extension_failedT_henning, rom_sizeT, extension_failedT, rom_two_scale, extension_failed,
                      verbose=False)

    times_printout(NWorldCoarse)

"""
#########################
 Errors
#########################
"""
if not certified_estimator_study:
    if use_fine_mesh:
        compute_errors(AFines, H1Fine, MFine, uFEMs, uLODcoarses, u_RBLODcoarses, u_TSRBLODcoarses,
                       u_resminTSRBLODcoarses, uLODfines, uRBLODfines, u_TSRBLODfines)
    else:
        H1Coarse = fem.assemblePatchMatrix(NWorldCoarse, world.ALocCoarse)
        MCoarse = fem.assemblePatchMatrix(NWorldCoarse, world.MLocCoarse)
        compute_coarse_errors(H1Coarse, MCoarse, xFulls, xFull_henning_RBs, u_TSRBLODs, u_resminTSRBLODs)

"""
########################################################################################################
                        Study the error behavior of the certified model
########################################################################################################
"""

from pymor.reductors.residual import ResidualOperator
m_two_scale_residual = ResidualOperator(reductor_two_scale.residual_operator, reductor_two_scale.residual_rhs)

if certified_estimator_study:
    coarse_estimator = reductor_two_scale.assemble_coarse_error_estimator()
    fine_estimator = reductor_two_scale.assemble_fine_error_estimator()
    full_estimator = reductor_two_scale.assemble_full_error_estimator()
    assert use_fine_mesh, "Does not work without fine mesh"
    print("\nEstimator study:")
    max_coarse_error, max_coarse_error_a, max_coarse_error_h1, max_full_error, max_full_error_h1 = 0, 0, 0, 0, 0
    max_eff, max_eff_h1, max_est, max_est_h1, max_fine_error, max_fine_error_h1 = 0, 0, 0, 0, 0, 0
    max_coarse_est_h1, max_coarse_est = 0, 0
    max_fine_est_h1, max_fine_est = 0, 0
    validation_set = training_set
    for mu_train in validation_set:
        print('.', end='', flush=True)
        # full LOD solution, see above
        KT, corT = zip(*pool.map(reference_solution, patchT,
                                 boundaryConditions=boundaryConditions, save_correctors=save_correctors,
                                 mu=mu_train, aFine_Constructor=aFine_Constructor, aFineCoefficients=aFineCoefficients))
        aFine = _construct_aFine_from_mu(aFines, aFineCoefficients, mu_train)
        if aFine.ndim == 1:
            ALocFine = world.ALocFine
        else:
            ALocFine = world.ALocMatrixFine
        AFine = fem.assemblePatchMatrix(NFine, ALocFine, aFine)
        H1Fine = fem.assemblePatchMatrix(NFine, world.ALocFine)

        KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KT)
        KFree = KFull[free][:, free]
        xFree = sparse.linalg.spsolve(KFree, bFree)
        xFull = np.zeros(world.NpCoarse)
        xFull[free] = xFree

        # TSRBLOD solution with reconstruction of corrector problem solutions
        U_ROM = rom_two_scale_optimized.solve(mu_train)
        u_rom = reductor_two_scale.reconstruct_partial(U_ROM)
        u_H_two_scale = np.zeros(world.NpCoarse)
        u_H_two_scale[free] = u_rom.to_numpy()[0]

        # without correctors
        u_LOD = basis * xFull
        u_TSRBLOD = basis * u_H_two_scale
        norm_fine = np.sqrt(np.dot((u_LOD), AFine * (u_LOD)))

        # with correctors
        corrector_errors, corrector_errors_h1, estimates = [], [], []
        u_tsrblod_full = reductor_two_scale.reconstruct(U_ROM)
        correctorsListT__ = [cor.to_numpy() for cor in u_tsrblod_full._blocks[1:]]

        basisCorrectors_true = pglod.assembleBasisCorrectors(world, patchT, corT)
        approx_corT = ()
        for cT_, basis_ in zip(correctorsListT__, basesT):
            approx_corT += (basis_.lincomb(cT_).to_numpy(),)

        utsf_s = []
        uhf_s = []

        for T, patch, product, h1product in zip(coarse_indices, patchT, productT, h1productT):
            from gridlod import util, coef
            # build energy product for mu
            loc_afine = coef.localizeCoefficient(patch, aFine)
            AFine_loc = fem.assemblePatchMatrix(patch.NPatchFine, ALocFine, loc_afine)

            NCoarseElement = world.NCoarseElement
            fineIndices, _ = util.fineIndicesInPatch(world.NWorldCoarse, NCoarseElement, patch.iPatchWorldCoarse,
                                                     patch.NPatchCoarse)
            c_true =  pglod.assembleBasisCorrectors(world, patchT, corT, T)
            c_full = (c_true * u_H_two_scale)[fineIndices]
            c_approx = approx_corT[T].T[:,0]
            c_diff = c_full - c_approx
            c_approx_full = np.zeros_like(u_LOD)
            c_approx_full[fineIndices] = c_approx.copy()
            utsf_s.append(c_approx_full)
            uhf_s.append(c_true * xFull)
            diff = product.source.from_numpy(c_diff.T)
            err = rho_sqrt * np.sqrt(np.dot(c_diff, AFine_loc * c_diff))
            err_h1 = rho_sqrt * np.sqrt(h1product.apply2(diff, diff))[0,0]
            corrector_errors.append(err)
            corrector_errors_h1.append(err_h1)

        u_LOD_with_f = u_LOD - sum(uhf_s)
        u_TSRBLOD_with_f = u_TSRBLOD - sum(utsf_s)
        u_LOD_with_f_true = (basis - basisCorrectors_true) * xFull

        energy_error_RB_with_f = np.sqrt(np.dot((u_TSRBLOD_with_f - u_LOD_with_f),
                                             AFine * (u_TSRBLOD_with_f - u_LOD_with_f)))
        energy_error_RB = np.sqrt(np.dot((u_TSRBLOD - u_LOD),
                                             AFine * (u_TSRBLOD - u_LOD)))
        h1_error_RB = np.sqrt(np.dot((u_TSRBLOD - u_LOD), H1Fine * (u_TSRBLOD - u_LOD)))

        full_error = corrector_errors.copy()
        full_error.append(energy_error_RB)
        full_error = np.linalg.norm(full_error)

        full_error_h1 = corrector_errors_h1.copy()
        full_error_h1.append(h1_error_RB)
        full_error_h1 = np.linalg.norm(full_error_h1)

        fine_error = np.linalg.norm(corrector_errors)
        fine_error_h1 = np.linalg.norm(corrector_errors_h1)

        u_train = rom_two_scale.solve(mu_train)
        full_estimate = full_estimator.estimate_error(u_train, mu_train, rom_two_scale)
        coarse_estimate = coarse_estimator.estimate_error(u_train, mu_train, rom_two_scale)
        fine_estimate = fine_estimator.estimate_error(u_train, mu_train, rom_two_scale)
        full_estimate_h1 = full_estimate * C_IH * 1/np.sqrt(min_alpha)
        coarse_estimate_h1 = coarse_estimate * C_IH * 1 / np.sqrt(min_alpha)
        fine_estimate_h1 = fine_estimate * C_IH * 1 / np.sqrt(min_alpha)
        eff = (full_estimate / full_error)[0]
        eff_h1 = (full_estimate_h1 / full_error_h1)[0]

        max_coarse_error = max(max_coarse_error, energy_error_RB_with_f)
        max_coarse_error_a = max(max_coarse_error_a, energy_error_RB)
        max_coarse_error_h1 = max(max_coarse_error_h1, h1_error_RB)
        
        max_fine_error = max(max_fine_error, fine_error)
        max_fine_error_h1 = max(max_fine_error_h1, fine_error_h1)

        max_full_error = max(max_full_error, full_error)
        max_full_error_h1 = max(max_full_error_h1, full_error_h1)

        max_est = max(max_est, full_estimate[0])
        max_est_h1 = max(max_est_h1, full_estimate_h1[0])
        
        max_eff = max(max_eff, eff)
        max_eff_h1 = max(max_eff_h1, eff_h1)

        max_coarse_est = max(max_coarse_est, coarse_estimate[0])
        max_coarse_est_h1 = max(max_coarse_est_h1, coarse_estimate_h1[0])

        max_fine_est = max(max_fine_est, fine_estimate[0])
        max_fine_est_h1 = max(max_fine_est_h1, fine_estimate_h1[0])

    print(f"\nmaximum full error          : {max_full_error:.9f}")
    print(f"maximum coarse error        : {max_coarse_error:.9f}")
    print(f"maximum estimator           : {max_est:.9f}")
    print(f"maximum full effectivity    : {max_eff:.9f}")
    print(f"maximum fine error          : {max_fine_error:.9f}")
    print(f"maximum fine estimator      : {max_fine_est:.9f}")
    print(f"maximum coarse estimator    : {max_coarse_est:.9f}")
    
    print(f"\nmaximum h1 full error       : {max_full_error_h1:.9f}")
    print(f"maximum h1 coarse error     : {max_coarse_error_h1:.9f}")
    print(f"maximum h1 estimator        : {max_est_h1:.9f}")
    print(f"maximum h1 full effectivity : {max_eff_h1:.9f}")
    print(f"maximum h1 fine error       : {max_fine_error_h1:.9f}")
    print(f"maximum h1 fine estimator   : {max_fine_est_h1:.9f}")
    print(f"maximum h1 coarse estimator : {max_coarse_est_h1:.9f}")
    print(f"maximum a  coarse error     : {max_coarse_error_a:.9f}")

    import os, csv
    # store in files
    def store_value(value, name, PATH="test_scripts/test_outputs/estimator_study"):
        arg = "a" if os.path.exists(f"{PATH}/{train_for}_{name}.txt") else "w"
        f = open(f"{PATH}/{train_for}_{name}.txt", arg)
        writer = csv.writer(f)
        writer.writerow([value])
        f.close()

    store_value(max_full_error, 'full_error')
    store_value(max_coarse_error, 'coarse_error')
    store_value(max_est, 'full_estimate')
    store_value(max_eff, 'effectivity')
    store_value(max_fine_error, 'fine_errpr')
    store_value(max_fine_est, 'fine_estimator')
    store_value(max_coarse_est, 'coarse_estimator')
    
    store_value(max_full_error_h1, 'full_error_h1')
    store_value(max_coarse_error_h1, 'coarse_error_h1')
    store_value(max_est_h1, 'full_estimate_h1')
    store_value(max_eff_h1, 'effectivity_h1')
    store_value(max_fine_error_h1, 'fine_error_h1')
    store_value(max_fine_est_h1, 'fine_estimator_h1')
    store_value(max_coarse_est_h1, 'coarse_estimator_h1')
    store_value(max_coarse_error_a, 'coarse_error_a')


del pool
