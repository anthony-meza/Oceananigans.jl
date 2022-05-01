using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions

N_iterations = 20 
N_ens = 10
rng_seed = 4137
Random.seed!(rng_seed)

prior_list = []
for a_i in 1:36
temp = Dict("distribution" => Parameterized(Normal(2, sqrt(2))), "constraint" => no_constraint(), "name" => "u"*string(a_i)
push!(prior_list, temp)
end

priors = ParameterDistribution(prior_list)
initial_ensemble = EKP.construct_initial_ensemble(priors, N_ensemble; rng_seed = rng_seed)
ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, G_target, Γ_stabilization, Inversion())

for i in 1:N_iterations     
params_i = get_u_final(ensemble_kalman_process)      
coeffs = Tuple(Tuple(params_i[i*j] for j in 1:6) for i in 1:6)
advection_schemes = [WENO5(vector_invariant = VorticityStencil(), smoothness_coeffs = coeffs 
#need to pass params_i into model somehow and generate some loss 
for Nh in [64]
…..
g_ens = hcat([G₁(params_i[:, i]) for i in 1:N_ensemble]...)      
EKP.update_ensemble!(ensemble_kalman_process, g_ens) 
end
end
