using DrWatson
@quickactivate "Polyhazard"
# For src
using LinearAlgebra, Distributions, Optim, Random, Combinatorics
# This script 
using Plots, CSV, DataFrames, JLD2

include(srcdir("Sampler.jl"))
include(srcdir("Postprocessing.jl"))

k = 1:4
sum(2 .^(1 .*k) .* binomial.(2 .+ k .- 1, k))

df = DataFrame(CSV.File(datadir("exp_pro","cost_c.csv")))
y = df[:,:time]
cens = convert(Array{Float64},df[:,:status])
covar = transpose([ones(size(y,1)) Matrix(df[:,2:14])])

Random.seed!(23463)
K0 = 2
x0 = rand(Normal(0,1),(size(covar,1)+1),K0)
v0 = rand(Bernoulli(0.5),size(x0,1), size(x0,2)).*2 .- 1.0
t0 = 0.0
dist0 = rand(DiscreteUniform(1,2),K0)
priors = PriorHyper(2.0,5.0,2.0,pdf.(Poisson(2),1:4),4,0.5,[1,2], 4,4, 2,1)
settings = Settings(100_000, 7_500_000, 50000.0, 50000.0, 50_000_000,50_000_000,10.0,5.0,2.0, 0.5, 0.9,0.5,0.01, 0.333, 0.5, 0.0,1.0)
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))
@time out1 = pdmp_sampler(x0, v0, t0, K0, dist0, dat, priors, settings, sample = true)
JLD2.save(datadir("sims","COST","COST_out1.jld2"),out1)

Random.seed!(982734)
K0 = 2
x0 = rand(Normal(0,1),(size(covar,1)+1),K0)
v0 = rand(Bernoulli(0.5),size(x0,1), size(x0,2)).*2 .- 1.0
t0 = 0.0
dist0 = rand(DiscreteUniform(1,2),K0)
priors = PriorHyper(2.0,5.0,2.0,pdf.(Poisson(2),1:4),4,0.5,[1,2], 4,4, 2,1)
settings = Settings(100_000, 7_500_000, 50000.0, 50000.0, 50_000_000,50_000_000,10.0,5.0,2.0, 0.5, 0.9,0.5,0.01, 0.333, 0.5, 0.0,1.0)
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))
@time out2 = pdmp_sampler(x0, v0, t0, K0, dist0, dat, priors, settings, sample = true)
JLD2.save(datadir("sims","COST","COST_out2.jld2"),out2)

Random.seed!(9019109)
K0 = 2
x0 = rand(Normal(0,1),(size(covar,1)+1),K0)
v0 = rand(Bernoulli(0.5),size(x0,1), size(x0,2)).*2 .- 1.0
t0 = 0.0
dist0 = rand(DiscreteUniform(1,2),K0)
priors = PriorHyper(2.0,5.0,2.0,pdf.(Poisson(2),1:4),4,0.5,[1,2], 4,4, 2,1)
settings = Settings(100_000, 7_500_000, 50000.0, 50000.0, 50_000_000,50_000_000,10.0,5.0,2.0, 0.5, 0.9,0.5,0.01, 0.333, 0.5, 0.0,1.0)
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))
@time out3 = pdmp_sampler(x0, v0, t0, K0, dist0, dat, priors, settings, sample = true)
JLD2.save(datadir("sims","COST","COST_out3.jld2"),out3)

Random.seed!(3463)
K0 = 1
x0 = rand(Normal(0,1),(size(covar,1)+1),K0)
v0 = rand(Bernoulli(0.5),size(x0,1), size(x0,2)).*2 .- 1.0
t0 = 0.0
dist0 = [1]
priors = PriorHyper(2.0,5.0,2.0,pdf.(Poisson(2),1:2),1,0.5,[1,2], 4,4, 2,1)
settings = Settings(40_000, 7_500_000, 20000.0, 20000.0, 50_000_000,50_000_000,10.0,5.0,2.0, 0.5, 0.9,0.5,0.01, 1.0, 0.5, 0.0,1.0)
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))
@time out1 = pdmp_sampler(x0, v0, t0, K0, dist0, dat, priors, settings, sample = true)
JLD2.save(datadir("sims","COST","COST_Weibull.jld2"),out1)
Random.seed!(8752)
K0 = 1
x0 = rand(Normal(0,1),(size(covar,1)+1),K0)
v0 = rand(Bernoulli(0.5),size(x0,1), size(x0,2)).*2 .- 1.0
t0 = 0.0
dist0 = [2]
priors = PriorHyper(2.0,5.0,2.0,pdf.(Poisson(2),1:2),1,0.5,[1,2], 4,4, 2,1)
settings = Settings(40_000, 7_500_000, 20000.0, 20000.0, 50_000_000,50_000_000,10.0,5.0,2.0, 0.5, 0.9,0.5,0.01, 1.0, 0.5, 0.0,1.0)
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))
@time out1 = pdmp_sampler(x0, v0, t0, K0, dist0, dat, priors, settings, sample = true)
JLD2.save(datadir("sims","COST","COST_LL.jld2"),out1)

out1 = JLD2.load(datadir("sims","COST","COST_out1.jld2"))
out2 = JLD2.load(datadir("sims","COST","COST_out2.jld2"))
out3 = JLD2.load(datadir("sims","COST","COST_out3.jld2"))
