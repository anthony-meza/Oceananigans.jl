using JLD2, Plots, Statistics 
cd("/home/ameza/batou-home/Repos/Oceananigans.jl/validation/bickley_jet/")
dir_names = ["G1_loss", "G2_loss", "G3_loss"]
title_names = [" G₁ loss ", " G₂ loss", " G₃ loss"]
for i in 1:length(dir_names)
    exp_name = dir_names[i]
    path = "/home/ameza/batou-home/Repos/Oceananigans.jl/validation/bickley_jet/"*exp_name*"/"
    filepath = path*exp_name * "_losses.jld2"
    x = load(filepath)
    saved_loss = x["single_stored_object"]
    title_name = title_names[i]
    (N_ensemble, N_iterations) = size(saved_loss)

    p1 = plot(title = title_name, legendfontsize=12, titlefont = 18,guidefontsize = 16,tickfont = 13,
    xlabel = "Iteration", ylabel = "Log Loss", legend=:bottomleft)  # empty Plot object
    
    for j in 1:N_ensemble
        # plot!(p1, (saved_loss[j, :] .- mean(saved_loss[j, :]))/std(saved_loss[j, :]), 
        # label = "Member " * string(j), linewidth = 2,)
        println(saved_loss[j, :])
        plot!(p1, log10.(saved_loss[j, :]), 
        label = "Member " * string(j), linewidth = 2,)
    end
    savefig(exp_name*".pdf")
end


