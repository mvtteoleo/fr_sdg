#include "ts_test.hpp"

/*
 *  The objective is to solve a simple non linear problem using different approaches.
 *
 *  The problem is u_dot = -Au
 *
 * A = [ 1 -1 0    |||           1
 *       0 1 -1    ||| u_ex = (  1 + t         ) * exp(-t)
 *       0 0 1 ]   |||           1 + t + 0.5t
 *
 *================================================================
 *
 *  The precondition option allows to solve instead the system by means of a
 *  change of variables:
 *
 *  u = H(u_tilde) = ln(u_tilde) || u_tilde = T(u) = exp(u)
 *
 *  The problem evolves to:
 *
 *  d H(u_tilde)/dt = -A H(u_tilde)
 *
 *================================================================
 *
 *  In order to make the problem as portable as possible I will only care
 *  about the RHSFunctions and RHSJacobian. This way PETSc can switch between
 *  Implicit and explicit time integration schemes by himself.
 *
 *  Also even if it comes with a minor penalty performance-wise the
 *  preconditioning in the case of the Implicit time in is handled by the
 *  TSSetTrans functions.
 *
 *================================================================
*  HOW TO RUN?
*  BDF_2
 ./build/test/ts_test  -ts_monitor_error -ts_dt 0.01 -ts_type bdf -ts_bdf_order 2 -ts_adapt_type
none

 * RK
 ./build/test/ts_test  -ts_monitor_error -ts_dt 0.1 -ts_type rk  -ts_adapt_type none
 */

int main(int argc, char** argv)
{
    // Initialize the problem
    numPDE::PETScData data(&argc, &argv);

    // Solve and write solution in phisical space
    data.solve(true);
    return 0;
}
