# Simulation Results for Non-Hermitian Rydberg-Atom Array

This repository contains the numerical simulation code and results presented in the following paper:

> **[Observation of Non-Hermitian Many-Body Phase Transition in a Rydberg-Atom Array](https://arxiv.org/abs/2512.02753)** 
>
> Yao-Wen Zhang, Biao Xu, Yijia Zhou, De-Sheng Xiang, Hao-Xiang Liu, Peng Zhou, Kuan Zhang, Ren Liao, Thomas Pohl, Weibin Li, Lin Li
>
> *arXiv preprint* [arXiv:2512.02753](https://arxiv.org/abs/2512.02753)

The simulations investigate PT-symmetry breaking and non-Hermitian many-body phenomena in a Rydberg-atom array. The main parameters are the coherent driving strength $\Omega$, dissipation rate $\Gamma$, dipolar exchange interaction $V$, and atom number $N$. The Loschmidt Echo (LE) measures the survival probability of the initial fully polarized state.

## Installation and Usage

Julia 1.12.5 is recommended; the project supports Julia 1.12.x. From the repository root, instantiate the locked environment with:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Start Jupyter Notebook in the same project environment with:

```bash
julia --project=. -e "using IJulia; notebook(dir=pwd())"
```

Open a notebook and select the Julia 1.12 kernel. The `data/` directory contains the precomputed data required by the plotting cells. Re-running the data-generation cells from scratch may take substantially longer and can start multiple parallel worker processes.

## energy-spectra.ipynb

Calculates the complex-energy spectrum, exceptional points (EPs), and many-body eigenmodes of the non-Hermitian Hamiltonian.

- **Figure 1(A) and Figure S6:** Show the evolution of the many-body spectrum with $V/\Gamma$ with the imaginary parts of the eigenvalues.

- **Figure 2(A,B) and Figure S5:** The real and imaginary parts of the two-atom spectrum.

- **Figure S14(D):** Compares multi-atom spin-wave energy levels under nearest-neighbor and long-range interactions.

## dynamics.ipynb

Simulates the Loschmidt Echo, many-body correlations, and non-Hermitian blockade dynamics. 

- **Figure 1(B) and Figure S15:** Calculate long-time-averaged two-body spin correlations.

- **Figure 2(C-H):** Simulate the two-atom LE and scaled LE, and compared with experiment results.

- **Figure 4(B) and Figure S14(E):** Examine the non-monotonic dependence of the LE on system size and compare the full, nearest-neighbor, and single-excitation models.

- **Figure 5(D,E) and Figure S13:** Demonstrate how interactions and dissipation jointly suppresses higher excitations.

- **Figure S7:** Compares Monte Carlo wave-function simulations with the effective non-Hermitian Schrödinger equation.

- **Figure S11:** Relates the normalized LE to the inverse participation ratio (IPR),.

## phase-diagram.ipynb

Determines the PT-symmetric and PT-symmetry-broken regions using the complex-energy spectrum and the Loschmidt rate function.

- **Figure 3 and Figures S8, S9:** Construct the two- and three-atom PT phase diagrams using the rate functions of the LE.

- **Figure 4(A) and Figure S10:** Present the multi-atom PT phase diagram and characterize the system-size dependence with both the rate function of LE and imaginary energy gap.
