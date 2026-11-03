/*************************************************************************************************
This code works only using the -snes_mf logic. This is hardcoded given the
Matshell J that allows to compute the Frèchlet derivative using the complex approach
./build/test/snes_claude_ex3  -snes_max_funcs unlimited -snes_max_it 1000000   -snes_view -ksp_view
*************************************************************************************************/
static char help[] = "Newton methods to solve u'' + u^{2} = f in parallel.\n\
Complex-step JFNK version: no explicit Jacobian is formed.\n\
J(u)*v is approximated as Im[ R(u + i*h*v) ] / h  with h = 1e-30.\n\
The command line options include:\n\
  -pre_check_iterates : activate checking of iterates\n\
  -post_check_iterates : activate checking of iterates\n\
  -check_tol <tol>: set tolerance for iterate checking\n\
  -user_precond : activate a (trivial) user-defined preconditioner\n\n";

#include <complex>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>
#include <vector>

/* ======================================================================
   Templated residual — works for T = double (SNES) and
                                    T = std::complex<double> (complex-step Jv)
   ====================================================================== */
template <typename T>
static void ComputeResidualLocal(PetscInt xs, PetscInt xm, PetscInt M, PetscReal mesh_h,
                                 const T* xx, // local array WITH ghost points
                                 const T* FF, // global F array (no ghosts, but xs-indexed)
                                 T*       ff)       // output residual, global-indexed owned points
{
    const T d = T(1.0 / (mesh_h * mesh_h));

    PetscInt xsi = xs, xei = xs + xm; // interior range (may shrink at boundaries)

    if (xs == 0)
    { // left boundary: u(0) = 0
        ff[0] = xx[0];
        xsi++;
    }
    if (xs + xm == M)
    { // right boundary: u(1) = 1
        ff[xs + xm - 1] = xx[xm - 1 + (xs == 0 ? 0 : 1)] - T(1.0);
        // Note: local index of global node xs+xm-1 is (xm-1) if no left ghost,
        //       or xm if there is a left ghost. Use ghost-offset arithmetic:
        PetscInt li     = (xs + xm - 1) - xs + (xs > 0 ? 1 : 0); // local idx with ghost
        ff[xs + xm - 1] = xx[li] - T(1.0);
        xei--;
    }

    // Interior: d*(u[i-1] - 2*u[i] + u[i+1]) + u[i]^2 - F[i]
    // Local index of global node i: li = i - xs + (xs>0 ? 1 : 0)
    //   (DMDA adds one ghost on the left if xs > 0)
    PetscInt ghost_offset = (xs > 0) ? 1 : 0;
    for (PetscInt i = xsi; i < xei; i++)
    {
        PetscInt li = i - xs + ghost_offset;
        ff[i]       = d * (xx[li - 1] - T(2.0) * xx[li] + xx[li + 1]) + xx[li] * xx[li] - FF[i];
    }
}

/* ======================================================================
   Application context
   ====================================================================== */
typedef struct
{
    DM          da;
    Vec         F; /* RHS of PDE */
    PetscMPIInt rank;
    PetscMPIInt size;
    PetscReal   h; /* mesh spacing */
} ApplicationCtx;

/* ======================================================================
   Context for the MatShell (complex-step Jv)
   ====================================================================== */
struct JvCtx
{
    ApplicationCtx* app;
    Vec             u_current; /* current Newton iterate (global) */
    Vec             u_local;   /* scratch local vec with ghosts (for u) */
    Vec             v_local;   /* scratch local vec with ghosts (for v) */
    double          cs_h;      /* complex-step size */
};

/* ======================================================================
   Monitoring context
   ====================================================================== */
typedef struct
{
    PetscViewer viewer;
} MonitorCtx;

/* ======================================================================
   Step-check context
   ====================================================================== */
typedef struct
{
    Vec             last_step;
    PetscReal       tolerance;
    ApplicationCtx* user;
} StepCheckCtx;

typedef struct
{
    PetscInt its0;
} SetSubKSPCtx;

/* ======================================================================
   Forward declarations
   ====================================================================== */
PetscErrorCode FormFunction(SNES, Vec, Vec, void*);
PetscErrorCode FormInitialGuess(Vec);
PetscErrorCode Monitor(SNES, PetscInt, PetscReal, void*);
PetscErrorCode PreCheck(SNESLineSearch, Vec, Vec, PetscBool*, void*);
PetscErrorCode PostCheck(SNESLineSearch, Vec, Vec, Vec, PetscBool*, PetscBool*, void*);
PetscErrorCode PostSetSubKSP(SNESLineSearch, Vec, Vec, Vec, PetscBool*, PetscBool*, void*);
PetscErrorCode MatrixFreePreconditioner(PC, Vec, Vec);

/* ======================================================================
   SNES residual callback  (T = double)
   ====================================================================== */
PetscErrorCode FormFunction(SNES snes, Vec x, Vec f, void* ctx)
{
    ApplicationCtx*    user = (ApplicationCtx*) ctx;
    DM                 da   = user->da;
    PetscScalar*       ff;
    const PetscScalar *xx, *FF;
    PetscInt           M, xs, xm;
    Vec                xlocal;

    PetscFunctionBeginUser;
    PetscCall(DMGetLocalVector(da, &xlocal));
    PetscCall(DMGlobalToLocalBegin(da, x, INSERT_VALUES, xlocal));
    PetscCall(DMGlobalToLocalEnd(da, x, INSERT_VALUES, xlocal));

    PetscCall(DMDAVecGetArrayRead(da, xlocal, (void**) &xx));
    PetscCall(DMDAVecGetArray(da, f, &ff));
    PetscCall(DMDAVecGetArrayRead(da, user->F, (void**) &FF));
    PetscCall(DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL));
    PetscCall(DMDAGetInfo(da, NULL, &M, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                          NULL));

    ComputeResidualLocal<PetscScalar>(xs, xm, M, user->h, xx, FF, ff);

    PetscCall(DMDAVecRestoreArrayRead(da, xlocal, (void**) &xx));
    PetscCall(DMDAVecRestoreArray(da, f, &ff));
    PetscCall(DMDAVecRestoreArrayRead(da, user->F, (void**) &FF));
    PetscCall(DMRestoreLocalVector(da, &xlocal));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/* ======================================================================
   MatMult for the complex-step MatShell:  w = J(u) * v
   ====================================================================== */
static PetscErrorCode MatMult_ComplexStep(Mat J, Vec v, Vec w)
{
    JvCtx*          ctx;
    ApplicationCtx* app;
    DM              da;
    PetscInt        M, xs, xm;

    PetscFunctionBeginUser;
    PetscCall(MatShellGetContext(J, &ctx));
    app = ctx->app;
    da  = app->da;

    PetscCall(DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL));
    PetscCall(DMDAGetInfo(da, NULL, &M, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                          NULL));

    // Scatter u_current and v to local vecs (with ghosts)
    PetscCall(DMGlobalToLocalBegin(da, ctx->u_current, INSERT_VALUES, ctx->u_local));
    PetscCall(DMGlobalToLocalEnd(da, ctx->u_current, INSERT_VALUES, ctx->u_local));
    PetscCall(DMGlobalToLocalBegin(da, v, INSERT_VALUES, ctx->v_local));
    PetscCall(DMGlobalToLocalEnd(da, v, INSERT_VALUES, ctx->v_local));

    const PetscScalar* pu_loc;
    const PetscScalar* pv_loc;
    const PetscScalar* pF;
    PetscScalar*       pw;

    PetscCall(DMDAVecGetArrayRead(da, ctx->u_local, (void**) &pu_loc));
    PetscCall(DMDAVecGetArrayRead(da, ctx->v_local, (void**) &pv_loc));
    PetscCall(DMDAVecGetArrayRead(da, app->F, (void**) &pF));
    PetscCall(DMDAVecGetArray(da, w, &pw));

    using cplx     = std::complex<double>;
    const double h = ctx->cs_h;

    // Get local size (with ghosts) from the local vec
    PetscInt local_size;
    PetscCall(VecGetLocalSize(ctx->u_local, &local_size));

    // Build complex u_c in local layout (includes ghosts)
    std::vector<cplx> u_c(local_size), R_c(xm);

    // Local ghost offset: DMDA 1-ghost stencil adds 1 ghost on the left
    // if xs > 0. The local array is indexed from xs-1 (or xs if boundary).
    // DMDAVecGetArray gives a pointer offset so that indexing by global i works.
    // So pu_loc[i] and pv_loc[i] are directly global-indexed.
    // The local_size covers the ghost range: (xs>0 ? xs-1 : 0) .. xs+xm-1+(xs+xm<M?1:0)
    // But since DMDAVecGetArray offsets the pointer, we access via global index.
    // We need to build u_c indexed the same way for passing to ComputeResidualLocal.

    // Determine ghost extent
    PetscInt gxs, gxm;
    PetscCall(DMDAGetGhostCorners(da, &gxs, NULL, NULL, &gxm, NULL, NULL));

    std::vector<cplx> u_c_ghost(gxm), F_c(xm);

    // Build complex perturbation over ghost range
    for (PetscInt i = gxs; i < gxs + gxm; i++)
        u_c_ghost[i - gxs] = cplx(pu_loc[i], h * pv_loc[i]);

    // Build F as complex (imaginary = 0, it's a constant RHS)
    for (PetscInt i = xs; i < xs + xm; i++)
        F_c[i - xs] = cplx(pF[i], 0.0);

    // Compute complex residual
    // ComputeResidualLocal expects: xx pointer offset to gxs, FF pointer offset to 0
    // We pass pointers offset so global indexing works
    std::vector<cplx> R_c_global(xs + xm); // only [xs..xs+xm-1] will be filled

    // Shift pointers to match global indexing (ComputeResidualLocal indexes by global i)
    const cplx* xx_ptr = u_c_ghost.data() - gxs; // xx[i] = u_c_ghost[i-gxs]
    const cplx* FF_ptr = F_c.data() - xs;        // FF[i] = F_c[i-xs]
    cplx*       ff_ptr = R_c_global.data();      // ff[i] direct global index

    ComputeResidualLocal<cplx>(xs, xm, M, app->h, xx_ptr, FF_ptr, ff_ptr);

    // Extract Jv = Im(R_c) / h
    for (PetscInt i = xs; i < xs + xm; i++)
        pw[i] = R_c_global[i].imag() / h;

    PetscCall(DMDAVecRestoreArrayRead(da, ctx->u_local, (void**) &pu_loc));
    PetscCall(DMDAVecRestoreArrayRead(da, ctx->v_local, (void**) &pv_loc));
    PetscCall(DMDAVecRestoreArrayRead(da, app->F, (void**) &pF));
    PetscCall(DMDAVecRestoreArray(da, w, &pw));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/* ======================================================================
   Jacobian callback: just update the current iterate pointer
   ====================================================================== */
static PetscErrorCode FormJacobian_Shell(SNES /*snes*/, Vec u, Mat J, Mat /*Jpre*/, void* ctx_void)
{
    JvCtx* ctx     = static_cast<JvCtx*>(ctx_void);
    ctx->u_current = u;
    (void) J;
    PetscFunctionReturn(PETSC_SUCCESS);
}

/* ======================================================================
   main
   ====================================================================== */
int main(int argc, char** argv)
{
    SNES           snes;
    SNESLineSearch linesearch;
    ApplicationCtx ctx;
    Vec            x, r, U, F;
    MonitorCtx     monP;
    StepCheckCtx   checkP;
    SetSubKSPCtx   checkP1;
    PetscBool      pre_check, post_check, post_setsubksp, flg;
    PetscScalar    xp, *FF, *UU, none = -1.0;
    PetscInt       its, N = 5, i, maxit, maxf, xs, xm;
    PetscReal      abstol, rtol, stol, norm;
    PetscBool      viewinitial = PETSC_FALSE;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &ctx.rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ctx.size));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-view_initial", &viewinitial, NULL));
    ctx.h = 1.0 / (N - 1);

    PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));

    // ── DMDA + vectors ──────────────────────────────────────────────
    PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, N, 1, 1, NULL, &ctx.da));
    PetscCall(DMSetFromOptions(ctx.da));
    PetscCall(DMSetUp(ctx.da));

    PetscCall(DMCreateGlobalVector(ctx.da, &x));
    PetscCall(VecDuplicate(x, &r));
    PetscCall(VecDuplicate(x, &F));
    ctx.F = F;
    PetscCall(VecDuplicate(x, &U));

    PetscCall(SNESSetFunction(snes, r, FormFunction, &ctx));

    // ── Complex-step MatShell ────────────────────────────────────────
    JvCtx jv_ctx;
    jv_ctx.app       = &ctx;
    jv_ctx.u_current = x;
    jv_ctx.cs_h      = 1e-13;

    // Local scratch vecs (with ghost points) for u and v scatters
    PetscCall(DMCreateLocalVector(ctx.da, &jv_ctx.u_local));
    PetscCall(DMCreateLocalVector(ctx.da, &jv_ctx.v_local));

    PetscInt n_local, n_global;
    PetscCall(VecGetLocalSize(x, &n_local));
    PetscCall(VecGetSize(x, &n_global));

    Mat J_shell;
    PetscCall(
        MatCreateShell(PETSC_COMM_WORLD, n_local, n_local, n_global, n_global, &jv_ctx, &J_shell));
    PetscCall(MatShellSetOperation(J_shell, MATOP_MULT, (void (*)(void)) MatMult_ComplexStep));

    PetscCall(SNESSetJacobian(snes, J_shell, J_shell, FormJacobian_Shell, &jv_ctx));

    // ── KSP / preconditioner ─────────────────────────────────────────
    PetscCall(PetscOptionsHasName(NULL, NULL, "-user_precond", &flg));
    if (flg)
    {
        KSP ksp;
        PC  pc;
        PetscCall(SNESGetKSP(snes, &ksp));
        PetscCall(KSPGetPC(ksp, &pc));
        PetscCall(PCSetType(pc, PCSHELL));
        PetscCall(PCShellSetApply(pc, MatrixFreePreconditioner));
    }
    else
    {
        KSP ksp;
        PC  pc;
        PetscCall(SNESGetKSP(snes, &ksp));
        PetscCall(KSPGetPC(ksp, &pc));
        PetscCall(PCSetType(pc, PCNONE)); // matrix-free: no preconditioner by default
    }

    // ── Monitor ──────────────────────────────────────────────────────
    PetscCall(PetscViewerDrawOpen(PETSC_COMM_WORLD, 0, 0, 0, 0, 400, 400, &monP.viewer));
    PetscCall(SNESMonitorSet(snes, Monitor, &monP, 0));
    PetscCall(PetscObjectSetName((PetscObject) x, "Approximate Solution"));
    PetscCall(PetscObjectSetName((PetscObject) U, "Exact Solution"));

    PetscCall(SNESSetFromOptions(snes));

    // ── Line-search callbacks ─────────────────────────────────────────
    PetscCall(SNESGetLineSearch(snes, &linesearch));

    PetscCall(PetscOptionsHasName(NULL, NULL, "-post_check_iterates", &post_check));
    if (post_check)
    {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Activating post step checking routine\n"));
        PetscCall(SNESLineSearchSetPostCheck(linesearch, PostCheck, &checkP));
        PetscCall(VecDuplicate(x, &checkP.last_step));
        checkP.tolerance = 1.0;
        checkP.user      = &ctx;
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-check_tol", &checkP.tolerance, NULL));
    }

    PetscCall(PetscOptionsHasName(NULL, NULL, "-post_setsubksp", &post_setsubksp));
    if (post_setsubksp)
    {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Activating post setsubksp\n"));
        PetscCall(SNESLineSearchSetPostCheck(linesearch, PostSetSubKSP, &checkP1));
    }

    PetscCall(PetscOptionsHasName(NULL, NULL, "-pre_check_iterates", &pre_check));
    if (pre_check)
    {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Activating pre step checking routine\n"));
        PetscCall(SNESLineSearchSetPreCheck(linesearch, PreCheck, &checkP));
    }

    PetscCall(SNESGetTolerances(snes, &abstol, &rtol, &stol, &maxit, &maxf));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "atol=%g, rtol=%g, stol=%g, maxit=%" PetscInt_FMT ", maxf=%" PetscInt_FMT
                          "\n",
                          (double) abstol, (double) rtol, (double) stol, maxit, maxf));

    // ── Initialize F and exact solution U ────────────────────────────
    PetscCall(DMDAGetCorners(ctx.da, &xs, NULL, NULL, &xm, NULL, NULL));
    PetscCall(DMDAVecGetArray(ctx.da, F, &FF));
    PetscCall(DMDAVecGetArray(ctx.da, U, &UU));
    xp = ctx.h * xs;
    for (i = xs; i < xs + xm; i++)
    {
        FF[i] = 6.0 * xp + PetscPowScalar(xp + 1.e-12, 6.0);
        UU[i] = xp * xp * xp;
        xp += ctx.h;
    }
    PetscCall(DMDAVecRestoreArray(ctx.da, F, &FF));
    PetscCall(DMDAVecRestoreArray(ctx.da, U, &UU));
    if (viewinitial)
    {
        PetscCall(VecView(U, 0));
        PetscCall(VecView(F, 0));
    }

    // ── Solve ─────────────────────────────────────────────────────────
    PetscCall(FormInitialGuess(x));
    PetscCall(SNESSolve(snes, NULL, x));
    PetscCall(SNESGetIterationNumber(snes, &its));
    PetscCall(
        PetscPrintf(PETSC_COMM_WORLD, "Number of SNES iterations = %" PetscInt_FMT "\n", its));

    // ── Error check ───────────────────────────────────────────────────
    PetscCall(VecAXPY(x, none, U));
    PetscCall(VecNorm(x, NORM_2, &norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Norm of error %g, Iterations %" PetscInt_FMT "\n",
                          (double) norm, its));

    // ── Clean up ──────────────────────────────────────────────────────
    PetscCall(PetscViewerDestroy(&monP.viewer));
    if (post_check) PetscCall(VecDestroy(&checkP.last_step));
    PetscCall(VecDestroy(&jv_ctx.u_local));
    PetscCall(VecDestroy(&jv_ctx.v_local));
    PetscCall(MatDestroy(&J_shell));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&r));
    PetscCall(VecDestroy(&U));
    PetscCall(VecDestroy(&F));
    PetscCall(SNESDestroy(&snes));
    PetscCall(DMDestroy(&ctx.da));
    PetscCall(PetscFinalize());
    return 0;
}

/* ======================================================================
   Boilerplate (unchanged from original)
   ====================================================================== */
PetscErrorCode FormInitialGuess(Vec x)
{
    PetscFunctionBeginUser;
    PetscCall(VecSet(x, 0.5));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Monitor(SNES snes, PetscInt its, PetscReal fnorm, void* ctx)
{
    MonitorCtx* monP = (MonitorCtx*) ctx;
    Vec         x;
    PetscFunctionBeginUser;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "iter = %" PetscInt_FMT ", SNES Function norm %g\n",
                          its, (double) fnorm));
    PetscCall(SNESGetSolution(snes, &x));
    PetscCall(VecView(x, monP->viewer));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PreCheck(SNESLineSearch /*ls*/, Vec /*xc*/, Vec /*y*/, PetscBool* changed_y,
                        void* /*ctx*/)
{
    PetscFunctionBeginUser;
    *changed_y = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PostCheck(SNESLineSearch linesearch, Vec /*xcurrent*/, Vec /*y*/, Vec x,
                         PetscBool* changed_y, PetscBool* changed_x, void* ctx)
{
    PetscInt        i, iter, xs, xm;
    StepCheckCtx*   check = (StepCheckCtx*) ctx;
    ApplicationCtx* user  = check->user;
    PetscScalar *   xa, *xa_last;
    PetscReal       rdiff;
    DM              da = user->da;
    SNES            snes;

    PetscFunctionBeginUser;
    *changed_x = PETSC_FALSE;
    *changed_y = PETSC_FALSE;
    PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
    PetscCall(SNESGetIterationNumber(snes, &iter));
    if (iter > 0)
    {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "Checking candidate step at iteration %" PetscInt_FMT
                              " with tolerance %g\n",
                              iter, (double) check->tolerance));
        PetscCall(DMDAVecGetArray(da, check->last_step, &xa_last));
        PetscCall(DMDAVecGetArray(da, x, &xa));
        PetscCall(DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL));
        for (i = xs; i < xs + xm; i++)
        {
            if (!PetscAbsScalar(xa[i]))
                rdiff = 2 * check->tolerance;
            else
                rdiff = PetscAbsScalar((xa[i] - xa_last[i]) / xa[i]);
            if (rdiff > check->tolerance)
            {
                PetscScalar tmp = xa[i];
                xa[i]           = .5 * (xa[i] + xa_last[i]);
                *changed_x      = PETSC_TRUE;
                PetscCall(PetscPrintf(
                    PETSC_COMM_WORLD,
                    "  Altering entry %" PetscInt_FMT ": x=%g, x_last=%g, diff=%g, x_new=%g\n", i,
                    (double) PetscAbsScalar(tmp), (double) PetscAbsScalar(xa_last[i]),
                    (double) rdiff, (double) PetscAbsScalar(xa[i])));
            }
        }
        PetscCall(DMDAVecRestoreArray(da, check->last_step, &xa_last));
        PetscCall(DMDAVecRestoreArray(da, x, &xa));
    }
    PetscCall(VecCopy(x, check->last_step));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PostSetSubKSP(SNESLineSearch linesearch, Vec /*xcurrent*/, Vec /*y*/, Vec /*x*/,
                             PetscBool* /*cy*/, PetscBool* /*cx*/, void* ctx)
{
    SetSubKSPCtx* check = (SetSubKSPCtx*) ctx;
    PetscInt      iter, its, sub_its, maxit;
    KSP           ksp, sub_ksp, *sub_ksps;
    PC            pc;
    PetscReal     ksp_ratio;
    SNES          snes;

    PetscFunctionBeginUser;
    PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
    PetscCall(SNESGetIterationNumber(snes, &iter));
    PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCBJacobiGetSubKSP(pc, NULL, NULL, &sub_ksps));
    sub_ksp = sub_ksps[0];
    PetscCall(KSPGetIterationNumber(ksp, &its));
    PetscCall(KSPGetIterationNumber(sub_ksp, &sub_its));
    if (iter)
    {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "    ...PostCheck snes iter %" PetscInt_FMT ", ksp_it %" PetscInt_FMT
                              " %" PetscInt_FMT ", subksp_it %" PetscInt_FMT "\n",
                              iter, check->its0, its, sub_its));
        ksp_ratio = (PetscReal) its / check->its0;
        maxit     = (PetscInt) (ksp_ratio * sub_its + 0.5);
        if (maxit < 2) maxit = 2;
        PetscCall(KSPSetTolerances(sub_ksp, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT, maxit));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                              "    ...ksp_ratio %g, new maxit %" PetscInt_FMT "\n\n",
                              (double) ksp_ratio, maxit));
    }
    check->its0 = its;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatrixFreePreconditioner(PC /*pc*/, Vec x, Vec y)
{
    PetscFunctionBeginUser;
    PetscCall(VecCopy(x, y));
    PetscFunctionReturn(PETSC_SUCCESS);
}
