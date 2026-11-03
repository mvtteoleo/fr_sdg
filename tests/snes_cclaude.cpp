/**
 * test.cpp — JFNK with complex-step Jacobian-vector product.
 *
 * Instead of the real finite-difference:
 *   J(u)·v ≈ [ R(u + ε·v) − R(u) ] / ε       (O(ε) accuracy, cancellation)
 *
 * we use the complex-step formula:
 *   J(u)·v ≈  Im[ R(u + i·h·v) ] / h          (O(h²) accuracy, no cancellation)
 *
 * h can be as small as 1e-30 without any subtraction cancellation, giving
 * near machine-precision Jacobian-vector products.
 *
 * Key design:
 *   - Residual is templated on the scalar type T, so it compiles for both
 *     T=double (used by SNES for function evaluations) and
 *     T=std::complex<double> (used inside the MatShell MatMult for Jv).
 *   - A MatShell wraps the complex-step logic and is handed to SNESSetJacobian.
 *   - No explicit Jacobian is formed anywhere.
 *
 * Problem :
 *   R_0(u) = u0^2 + u1^2 - 1 = 0
 *   R_1(u) = u0  - u1        = 0
 *   Solution: u0 = u1 = 1/sqrt(2) ≈ 0.7071
 *
 * Run:
 *   ./test
 *   ./test -snes_monitor -ksp_monitor -snes_view
 */

#include <complex>
#include <petscsnes.h>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Residual templated on scalar type T.
// Works for T = double  (PETSc function evaluation)
//          T = std::complex<double>  (complex-step Jv)
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
void residual(const T* u, T* R)
{
    R[0] = u[0] * u[0] + u[1] * u[1] - T(1); // u0^2 + u1^2 - 1
    R[1] = u[0] - u[1];                      // u0  - u1
}

// ─────────────────────────────────────────────────────────────────────────────
// PETSc residual callback (T = double)
// ─────────────────────────────────────────────────────────────────────────────
static PetscErrorCode FormResidual(SNES /*snes*/, Vec u, Vec F, void* /*ctx*/)
{
    const PetscScalar* pu;
    PetscScalar*       pF;
    PetscCall(VecGetArrayRead(u, &pu));
    PetscCall(VecGetArray(F, &pF));
    residual(pu, pF);
    PetscCall(VecRestoreArrayRead(u, &pu));
    PetscCall(VecRestoreArray(F, &pF));
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Context passed to the MatShell
// ─────────────────────────────────────────────────────────────────────────────
struct JvCtx
{
    Vec    u_current; // current Newton iterate (real, owned by SNES)
    double h;         // complex-step step size
};

// ─────────────────────────────────────────────────────────────────────────────
// MatMult for the MatShell: computes w = J(u)·v via complex step
//
//   u_c[i] = u[i] + i * h * v[i]          (perturb in imaginary direction)
//   R_c    = R(u_c)                        (complex residual evaluation)
//   w[i]   = Im( R_c[i] ) / h             (extract Jv)
// ─────────────────────────────────────────────────────────────────────────────
static PetscErrorCode MatMult_ComplexStep(Mat J, Vec v, Vec w)
{
    JvCtx* ctx;
    PetscCall(MatShellGetContext(J, &ctx));

    const PetscInt N = 2;

    const PetscScalar* pu;
    const PetscScalar* pv;
    PetscScalar*       pw;
    PetscCall(VecGetArrayRead(ctx->u_current, &pu));
    PetscCall(VecGetArrayRead(v, &pv));
    PetscCall(VecGetArray(w, &pw));

    using cplx = std::complex<double>;

    // Build complex input: u + i*h*v
    std::vector<cplx> u_c(N), R_c(N);
    for (PetscInt i = 0; i < N; ++i)
        u_c[i] = cplx(pu[i], ctx->h * pv[i]);

    // Evaluate residual with complex arithmetic
    residual(u_c.data(), R_c.data());

    // Jv ≈ Im(R(u + i*h*v)) / h
    for (PetscInt i = 0; i < N; ++i)
        pw[i] = R_c[i].imag() / ctx->h;

    PetscCall(VecRestoreArrayRead(ctx->u_current, &pu));
    PetscCall(VecRestoreArrayRead(v, &pv));
    PetscCall(VecRestoreArray(w, &pw));
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Jacobian callback: update the MatShell's current iterate pointer,
// then return the same shell as both the operator and preconditioning matrix.
// ─────────────────────────────────────────────────────────────────────────────
static PetscErrorCode FormJacobian(SNES /*snes*/, Vec u, Mat J, Mat /*Jpre*/, void* ctx_void)
{
    JvCtx* ctx     = static_cast<JvCtx*>(ctx_void);
    ctx->u_current = u; // point to the current iterate (no copy needed)
    (void) J;
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));

    const PetscInt N = 2;

    // ── Vectors ────────────────────────────────────────────────────
    Vec u, F;
    PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
    PetscCall(VecSetSizes(u, PETSC_DECIDE, N));
    PetscCall(VecSetFromOptions(u));
    PetscCall(VecDuplicate(u, &F));

    // Initial guess
    PetscCall(VecSet(u, 0.5));

    // ── Complex-step MatShell ──────────────────────────────────────
    JvCtx jv_ctx;
    jv_ctx.u_current = u;
    jv_ctx.h         = 1e-13; // can be extremely small: no cancellation

    Mat J_shell;
    PetscCall(MatCreateShell(PETSC_COMM_WORLD, N, N, N, N, &jv_ctx, &J_shell));
    PetscCall(MatShellSetOperation(J_shell, MATOP_MULT, (void (*)(void)) MatMult_ComplexStep));

    // ── SNES ───────────────────────────────────────────────────────
    SNES snes;
    PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
    PetscCall(SNESSetFunction(snes, F, FormResidual, nullptr));
    PetscCall(SNESSetJacobian(snes, J_shell, J_shell, FormJacobian, &jv_ctx));

    // ── KSP / preconditioner ───────────────────────────────────────
    KSP ksp;
    PetscCall(SNESGetKSP(snes, &ksp));
    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCNONE));
    PetscCall(KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, 200));

    PetscCall(SNESSetFromOptions(snes));

    // ── Solve ──────────────────────────────────────────────────────
    PetscCall(SNESSolve(snes, nullptr, u));

    // ── Report ─────────────────────────────────────────────────────
    SNESConvergedReason reason;
    PetscInt            iters;
    PetscCall(SNESGetConvergedReason(snes, &reason));
    PetscCall(SNESGetIterationNumber(snes, &iters));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nConverged: %s  (%d Newton iterations)\n",
                          SNESConvergedReasons[reason], (int) iters));

    const PetscScalar* pu;
    PetscCall(VecGetArrayRead(u, &pu));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "Solution:  u0 = %.15g\n"
                          "           u1 = %.15g\n"
                          "     exact u* = %.15g\n",
                          (double) pu[0], (double) pu[1], 1.0 / PetscSqrtReal(2.0)));
    PetscCall(VecRestoreArrayRead(u, &pu));

    // ── Clean up ───────────────────────────────────────────────────
    PetscCall(MatDestroy(&J_shell));
    PetscCall(VecDestroy(&u));
    PetscCall(VecDestroy(&F));
    PetscCall(SNESDestroy(&snes));
    PetscCall(PetscFinalize());
    return 0;
}
