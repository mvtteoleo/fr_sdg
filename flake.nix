 {
  description = "Dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        mergedYaml = pkgs.symlinkJoin {
          name = "libyaml-merged";
          paths = [ pkgs.libyaml.dev pkgs.libyaml.out ];
        };

        pyEnv = pkgs.python3.withPackages (ps: with ps; [
          scipy
          sympy
          pandas
          matplotlib
          numpy
          black
         openpyxl
        ]);

        # 2. Override the PETSc derivation
        myPetscDeriv = pkgs.petsc;


      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Version control
            pkgs.git

            # Compilers
            pkgs.gnumake
            pkgs.gcc15
            pkgs.clang-tools
            pkgs.openmpi

            # Libraries
            pkgs.ginac
            pkgs.fftw
            pkgs.eigen
            pkgs.boost
            pkgs.gnuplot
            pkgs.tbb

            # Tools
            pkgs.heaptrack
            pkgs.gdb
            pkgs.neovim

            pyEnv
            myPetscDeriv
          ];

          shellHook = ''
            # Set up environment variables
            export EIGEN_INCLUDE_DIR="${pkgs.eigen}/include/eigen3"
            export FFTW_INCLUDE_DIR="${pkgs.fftw}/include"
            export PETSC_DIR="${myPetscDeriv}"
            export PETSC_ARCH=""

            # Just unset LD to avoid casual linkings
            unset LD_LIBRARY_PATH
		

            echo "✅ Thesis environment activated"
          '';
        };
      }
    );
}
