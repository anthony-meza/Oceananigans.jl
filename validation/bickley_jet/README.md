
# bickley_jet 
This example simulates the evolution of an unstable, geostrophically balanced, Bickley jet. The initial conditions superpose the Bickley jet with small-amplitude perturbations.See "The nonlinear evolution of barotropically unstable jets," J. Phys. Oceanogr. (2003) for more details on this problem. This example also uses the WENO5 advection scheme with smoothness coefficients that are optimized with respect to an arbitrary loss function **G**. The  method used to optimize the coefficients is Ensemble Kalman Inversion (EKI).  

## Installation and usage
1) Clone EnsembleKalmanProcesses to local machine (`https://github.com/CliMA/EnsembleKalmanProcesses.jl `)
2) From the Julia REPL: `using Pkg; Pkg.add(url = "/path/to/EnsembleKalmanProcesses.jl/")`
3) Run bickley_jet_WENO_with_EKI.jl in REPL or in terminal. Feel free to choose a loss function (G1, G2, G3) or choose your own.  


## Dependencies
- EnsembleKalmanProcesses (must clone to local machine) 
