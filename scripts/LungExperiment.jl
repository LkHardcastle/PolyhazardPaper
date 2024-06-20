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
covar = transpose([ones(size(y,1)) df[:,:SLT] .- 0.5])

Random.seed!(34632)
s1 = []
s2 = []
tm = []
p1 = [25.0,10.0,5.0,2.0]
K0 = 2
x0 = rand(Normal(0,1),(size(covar,1)+1),K0)
v0 = rand(Bernoulli(0.5),size(x0,1), size(x0,2)).*2 .- 1.0
t0 = 0.0
dist0 = [1,1]
settings = Settings(40_000, 1_000_000, 20000.0, 20000.0, 20_000_000,20_000_000,0.0,5.0,2.0, 0.5, 0.9,0.5,0.01, 0.0, 0.5, 0.0,1.0)
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))
ind = 1
for i in eachindex(p1)
    #priors = PriorHyper(2.0,p1[i],p1[i],pdf.(Poisson(2),1:2),2,0.1,[1,2], 4,1, 100,1)
    priors =  PriorHyper(2.0,p1[i],p1[i],pdf.(Poisson(2),1:2),4,0.1,[1,2], 4,1, 2,1)
    println(priors)
    out = pdmp_sampler(x0, v0, t0, K0, dist0, dat, priors, settings, sample = true)
    JLD2.save(datadir("sims","Lung_exp_3","out$ind.jld2"),out)
    ind += 1
end

out = JLD2.load(datadir("sims","Lung_exp_3","out10.jld2"))
out1 = JLD2.load(datadir("sims","Lung_exp_3","out11.jld2"))