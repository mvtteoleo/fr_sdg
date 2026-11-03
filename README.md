This project is a solver for Euler equations, able to handle generic gas models
(Ideal and Van der Waals as of now, but extensibility is key).

The numerical framework is that of Flux Reconstruction scheme equivalent to
Spectral Discontinuous Galerkin.

Both implicit and explicit time integration is available by means of PETSc,
which handle the backhand for the time stepping and nonlinear solver.
A note of interest goes for the possibility of using complex step derivative to
approximate the Jacobian.

The C++ standard of choice is C++23, to manage the dependencies I here use nix,
the only real dependency.

If one is missing nix can avoid installing it globally using [nix-portable](https://github.com/DavHau/nix-portable).

The [flake.nix](./flake.nix) file is the one that handles dependencies and packages needed.

To run one needs to either call `nix develop` or `./nix-portable nix develop
--extra-experimental-features "nix-command flakes"` to enter in the
nix-shell. 

The tests are quite a few to both show the usage of some more advanced and
modern C++ features such as views and spans, and also for ease of integration.

All the executables will be in the build directory. 

Once inside the nix-shell is enough to call 
`make all` to build all the executables `make <TEST-NAME>` will build `tests/<TEST-NAME>.cpp` .


One note on the tools/ directory, it contains some .py to handle
solutions by means of method of characteristics and in general helpers to
automate runs.

