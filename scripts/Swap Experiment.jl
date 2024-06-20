using DrWatson
@quickactivate "Polyhazard"
# For src
using LinearAlgebra, Distributions, Optim, Random, Combinatorics
# This script 
using Plots, CSV, DataFrames, JLD2

include(srcdir("Sampler.jl"))
include(srcdir("Postprocessing.jl"))

Random.seed!(2222)
y_t = min(exp.(rand(Normal(0,0.5), 100)),rand(Weibull(1,1), 100))
cens_t = rand(Exponential(0.5),100)
y = y_t.*(y_t .== min.(y_t, cens_t)) .+ cens_t.*(y_t .!= min.(y_t, cens_t))
cens = y_t .== min.(y_t, cens_t) .+ 0.0
covar = transpose([ones(100) rand(Bernoulli(0.5),100)])


Random.seed!(141)
K0 = 2
x0 = rand(Normal(0,1),(size(covar,1)+1),K0)
v0 = rand(Bernoulli(0.5),size(x0,1), size(x0,2)).*2 .- 1.0
t0 = 0.0
dist0 = rand(DiscreteUniform(1,2),K0)
priors = PriorHyper(2.0,5.0,5.0,pdf.(Poisson(2),1:4),4,0.5,[1,2], 4,1, 2,1)
settings = Settings(20_000, 500_000, 20000.0, 20000.0, 20_000_000,20_000_000,10.0,5.0,1.0, 0.0, 0.9,0.5,0.01, 0.0, 0.5, 0.0,0.0)
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))
@time out1 = pdmp_sampler(x0, v0, t0, K0, dist0, dat, priors, settings, sample = true)
JLD2.save(datadir("sims","SwapSim","out11.jld2"),out1)
@time out1 = pdmp_sampler(x0, v0, t0, K0, dist0, dat, priors, settings, sample = true)
JLD2.save(datadir("sims","SwapSim","out12.jld2"),out1)

Random.seed!(978)
K0 = 2
x0 = rand(Normal(0,1),(size(covar,1)+1),K0)
v0 = rand(Bernoulli(0.5),size(x0,1), size(x0,2)).*2 .- 1.0
t0 = 0.0
dist0 = rand(DiscreteUniform(1,2),K0)
priors = PriorHyper(2.0,5.0,5.0,pdf.(Poisson(2),1:4),4,0.5,[1,2], 4,1, 2,1)
settings = Settings(20_000, 500_000, 20000.0, 20000.0, 20_000_000,20_000_000,10.0,5.0,1.0, 0.5, 0.9,0.5,0.01, 0.0, 0.5, 1.0,0.0)
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))
@time out2 = pdmp_sampler(x0, v0, t0, K0, dist0, dat, priors, settings, sample = true)
JLD2.save(datadir("sims","SwapSim","out21.jld2"),out2)
@time out2 = pdmp_sampler(x0, v0, t0, K0, dist0, dat, priors, settings, sample = true)
JLD2.save(datadir("sims","SwapSim","out22.jld2"),out2)

Random.seed!(253)
K0 = 2
x0 = rand(Normal(0,1),(size(covar,1)+1),K0)
v0 = rand(Bernoulli(0.5),size(x0,1), size(x0,2)).*2 .- 1.0
t0 = 0.0
dist0 = [1,1]
priors = PriorHyper(2.0,5.0,5.0,pdf.(Poisson(2),1:4),4,0.5,[1,2], 4,1, 2,1)
settings = Settings(20_000, 500_000, 20000.0, 20000.0, 20_000_000,20_000_000,10.0,5.0,1.0, 0.5, 0.9,0.5,0.01, 0.0, 0.5, 0.0,1.0)
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))
@time out3 = pdmp_sampler(x0, v0, t0, K0, dist0, dat, priors, settings, sample = true)
JLD2.save(datadir("sims","SwapSim","out31.jld2"),out3)
@time out3 = pdmp_sampler(x0, v0, t0, K0, dist0, dat, priors, settings, sample = true)
JLD2.save(datadir("sims","SwapSim","out32.jld2"),out3)

out_p1 = posterior_probs_smps(out1, [1,2])
out_p2 = posterior_probs_smps(out2, [1,2])
out_p3 = posterior_probs_smps(out3, [1,2])

ind = union(findall(out_p1["Dist_probs"] .> 0.05),findall(out_p2["Dist_probs"] .> 0.05),findall(out_p3["Dist_probs"] .> 0.05))

plot(transpose(dist_roll(out1, [1,2]))[1:5_000,ind], legend = false)
plot(transpose(dist_roll(out2, [1,2]))[1:5_000,ind], legend = false)
plot(transpose(dist_roll(out3, [1,2]))[1:5_000,ind], legend = false)