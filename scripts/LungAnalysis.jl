using DrWatson
@quickactivate "PolyhazardPaper"
# For src
using LinearAlgebra, Distributions, Optim, Random, Combinatorics
# This script 
using Plots, CSV, DataFrames, JLD2

include(srcdir("Sampler.jl"))
include(srcdir("Postprocessing.jl"))

df = DataFrame(CSV.File(datadir("exp_raw","DigLung.csv")))
y = df[:,:time]
cens = convert(Array{Float64},df[:,:event])
covar = transpose([ones(size(y,1)) df[:,:SLT] - 0.5])

Random.seed!(2352)
K0 = 2
x0 = rand(Normal(0,1),(size(covar,1)+1),K0)
v0 = rand(Bernoulli(0.5),size(x0,1), size(x0,2)).*2 .- 1.0
t0 = 0.0
dist0 = rand(DiscreteUniform(1,2),K0)
priors = PriorHyper(2.0,5.0,5.0,pdf.(Poisson(2),1:4),4,0.5,[1,2], 4,1, 2,1)
settings = Settings(80_000, 500_000, 10000.0, 10000.0, 20_000_000,20_000_000,10.0,5.0,4.0, 0.5, 0.9,0.5,0.01, 0.0, 0.5, 0.0,1.0)
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))
@time out1 = pdmp_sampler(x0, v0, t0, K0, dist0, dat, priors, settings, sample = true)
JLD2.save(datadir("sims","Dem_dig","Dem_out1.jld2"),out1)



Random.seed!(14124)
K0 = 2
x0 = rand(Normal(0,1),(size(covar,1)+1),K0)
v0 = rand(Bernoulli(0.5),size(x0,1), size(x0,2)).*2 .- 1.0
t0 = 0.0
dist0 = rand(DiscreteUniform(1,2),K0)
priors = PriorHyper(2.0,5.0,5.0,pdf.(Poisson(2),1:4),4,0.5,[1,2], 4,1, 2,1)
settings = Settings(80_000, 500_000, 20000.0, 20000.0, 20_000_000,20_000_000,10.0,5.0,4.0, 0.5, 0.9,0.5,0.01, 0.0, 0.5, 0.0,1.0)
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))
@time out2 = pdmp_sampler(x0, v0, t0, K0, dist0, dat, priors, settings, sample = true)
JLD2.save(datadir("sims","Dem_dig","Dem_out2.jld2"),out2)
