using DrWatson
@quickactivate "Polyhazard"
# For src
using LinearAlgebra, Distributions, Optim, Random, Combinatorics
# This script 
using Plots, CSV, DataFrames, JLD2, SurvivalAnalysis

include(srcdir("Sampler.jl"))
include(srcdir("Postprocessing.jl"))
include(srcdir("EDA.jl"))

df = DataFrame(CSV.File(datadir("exp_pro","KT_data.csv")))
y = df[:,:time]
cens = convert(Array{Float64},df[:,:event])
covar = transpose([ones(size(y,1)) df[:,:Hypertension] df[:,:sex] df[:,:Dyslipidemia] df[:,:age2] df[:,:age3] df[:,:age4] df[:,:age5] df[:,:age6] df[:,:age7] df[:,:wait2] df[:,:wait3] df[:,:wait4]])
covar[2:12,:] = covar[2:12,:] .- 0.5

mean(cens[intersect(findall(df[:,:age2] .== 0.0),findall(df[:,:age3] .== 0.0),findall(df[:,:age4] .== 0.0),findall(df[:,:age5] .== 0.0),findall(df[:,:age6] .== 0.0),findall(df[:,:age7] .== 0.0))])
mean(cens[findall(df[:,:age2] .== 1.0)])
mean(cens[findall(df[:,:age6] .== 1.0)])
mean(cens[findall(df[:,:age7] .== 1.0)])
mean(cens[intersect(findall(df[:,:wait2] .== 0.0),findall(df[:,:wait3] .== 0.0),findall(df[:,:wait4] .== 0.0))])
mean(cens[findall(df[:,:wait2] .== 1.0)])
mean(cens[findall(df[:,:wait3] .== 1.0)])
mean(cens[findall(df[:,:wait4] .== 1.0)])

sum(df[:,:age7] .== 1.0)
sum(df[:,:wait2] .== 1.0)
sum(df[:,:wait3] .== 1.0)
sum(df[:,:wait4] .== 1.0)

Random.seed!(7142948)
K0 = 2
x0 = rand(Normal(0,1),(size(covar,1)+1),K0)
v0 = rand(Bernoulli(0.5),size(x0,1), size(x0,2)).*2 .- 1.0
t0 = 0.0
dist0 = [1,1]
priors = PriorHyper(1.0,5.0,2.0,pdf.(Poisson(2),1:3),3,0.5,[1,2], 4,4, 2,1)
settings = Settings(200_000, 4_000_000, 20000.0, 20000.0, 40_000_000,40_000_000,20.0,5.0,10.0, 0.5, 0.9,0.5,0.01, 0.333, 0.5, 0.0,1.0)
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))
@time out1 = pdmp_sampler(x0, v0, t0, K0, dist0, dat, priors, settings, sample = true)
JLD2.save(datadir("sims","TKT","TKT_out1.jld2"),out1)

Random.seed!(2346236)
K0 = 2
x0 = rand(Normal(0,1),(size(covar,1)+1),K0)
v0 = rand(Bernoulli(0.5),size(x0,1), size(x0,2)).*2 .- 1.0
t0 = 0.0
dist0 = [2,2]
priors = PriorHyper(1.0,5.0,2.0,pdf.(Poisson(2),1:3),3,0.5,[1,2], 4,4, 2,1)
settings = Settings(200_000, 4_000_000, 20000.0, 20000.0, 40_000_000,40_000_000,20.0,5.0,10.0, 0.5, 0.9,0.5,0.01, 0.333, 0.5, 0.0,1.0)
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))
@time out2 = pdmp_sampler(x0, v0, t0, K0, dist0, dat, priors, settings, sample = true)
JLD2.save(datadir("sims","TKT","TKT_out2.jld2"),out2)

out1 = JLD2.load(datadir("sims","TKT","TKT_out1.jld2"))
out2 = JLD2.load(datadir("sims","TKT","TKT_out2.jld2"))


Random.seed!(8787434)
K0 = 1
x0 = rand(Normal(0,1),(size(covar,1)+1),K0)
v0 = rand(Bernoulli(0.5),size(x0,1), size(x0,2)).*2 .- 1.0
t0 = 0.0
dist0 = [1]
priors = PriorHyper(1.0,5.0,2.0,pdf.(Poisson(2),1:3),1,0.5,[1,2], 4,4, 2,1)
settings = Settings(100_000, 1_500_000, 10000.0, 10000.0, 40_000_000,40_000_000,20.0,5.0,10.0, 0.5, 0.9,0.5,0.01, 1.0, 0.5, 0.0,1.0)
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))
@time out3 = pdmp_sampler(x0, v0, t0, K0, dist0, dat, priors, settings, sample = true)
JLD2.save(datadir("sims","TKT","TKT_outWW.jld2"),out3)


out1 = JLD2.load(datadir("sims","TKT","TKT_out1.jld2"))
out2 = JLD2.load(datadir("sims","TKT","TKT_out2.jld2"))

out1["Eval"]
out_p1 = posterior_probs_smps(out1, [1,2])
out2["Eval"]
out_p2 = posterior_probs_smps(out2, [1,2])
out1["Eval"].births/out1["Eval"].b_attempt
out1["Eval"].deaths/out1["Eval"].d_attempt
out1["Eval"].swaps1/out1["Eval"].s_attempt1
plot(transpose(K_roll(out1)))
plot!(transpose(K_roll(out2)))
plot(transpose(dist_roll(out1, [1,2]))[:,findall(out_p1["Dist_probs"] .> 0.01)])
plot!(transpose(dist_roll(out2, [1,2]))[:,findall(out_p1["Dist_probs"] .> 0.01)])
heat_smp_plot(out_p1, out1)
VS_out1 = VS_out(sort_out(out1, [1,2], dat),dat)
VS_out1[:,:,findall(out_p1["Dist_probs"] .> 0.01)]
s_out = sort_out(out1,[1,2],dat)

trunc.(mean(hcat(out_p1["Dist_probs"],out_p1["Dist_probs"]), dims = 2),digits = 3)

t = 0.01:0.05:20
cov = vcat(zeros(4),1,zeros(4),zeros(3))
haz1_, haz1, Surv1 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[10] = 1
haz1_, haz2, Surv2 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[10] = 0
cov[11] = 1
haz1_, haz3, Surv3 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[11] = 0
cov[12] = 1
haz1_, haz4, Surv4 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
plot(t,haz1)
plot!(t,haz2)
plot!(t,haz3)
plot!(t,haz4)

t = 0.01:0.05:20
cov = vcat(zeros(8),0,zeros(3))
haz1_, haz1, Surv1 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[4] = 1
haz1_, haz2, Surv2 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[4] = 0
cov[5] = 1
haz1_, haz3, Surv3 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[5] = 0
cov[6] = 1
haz1_, haz4, Surv4 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[6] = 0
cov[7] = 1
haz1_, haz5, Surv5 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[7] = 0
cov[8] = 1
haz1_, haz6, Surv6 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[8] = 0
cov[9] = 1
haz1_, haz7, Surv7 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
plot(t,haz1)
plot!(t,haz2)
plot!(t,haz3)
plot!(t,haz4)
plot!(t,haz5)
plot!(t,haz6)
plot!(t,haz7)

plot(t,Surv1, ylimits = (0,1))
plot!(t,Surv2)
plot!(t,Surv3)
plot!(t,Surv4)
plot!(t,Surv5)
plot!(t,Surv6)
plot!(t,Surv7)
savefig(plotsdir("TKT","Surv1.png"))

t = 0.01:0.05:20
cov = vcat(zeros(8),0,1,zeros(2))
haz1_, haz1, Surv1 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[4] = 1
haz1_, haz2, Surv2 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[4] = 0
cov[5] = 1
haz1_, haz3, Surv3 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[5] = 0
cov[6] = 1
haz1_, haz4, Surv4 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[6] = 0
cov[7] = 1
haz1_, haz5, Surv5 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[7] = 0
cov[8] = 1
haz1_, haz6, Surv6 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[8] = 0
cov[9] = 1
haz1_, haz7, Surv7 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)

plot(t,haz1)
plot!(t,haz2)
plot!(t,haz3)
plot!(t,haz4)
plot!(t,haz5)
plot!(t,haz6)
plot!(t,haz7)

plot(t,Surv1, ylimits = (0,1))
plot!(t,Surv2)
plot!(t,Surv3)
plot!(t,Surv4)
plot!(t,Surv5)
plot!(t,Surv6)
plot!(t,Surv7)
savefig(plotsdir("TKT","Surv2.png"))

t = 0.01:0.05:20
cov = vcat(zeros(8),0,0,1,0)
haz1_, haz1, Surv1 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[4] = 1
haz1_, haz2, Surv2 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[4] = 0
cov[5] = 1
haz1_, haz3, Surv3 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[5] = 0
cov[6] = 1
haz1_, haz4, Surv4 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[6] = 0
cov[7] = 1
haz1_, haz5, Surv5 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[7] = 0
cov[8] = 1
haz1_, haz6, Surv6 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[8] = 0
cov[9] = 1
haz1_, haz7, Surv7 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
plot(t,haz1)
plot!(t,haz2)
plot!(t,haz3)
plot!(t,haz4)
plot!(t,haz5)
plot!(t,haz6)
plot!(t,haz7)

plot(t,Surv1, ylimits = (0,1))
plot!(t,Surv2)
plot!(t,Surv3)
plot!(t,Surv4)
plot!(t,Surv5)
plot!(t,Surv6)
plot!(t,Surv7)
savefig(plotsdir("TKT","Surv3.png"))


t = 0.01:0.05:20
cov = vcat(zeros(8),0,0,0,1)
haz1_, haz1, Surv1 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[4] = 1
haz1_, haz2, Surv2 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[4] = 0
cov[5] = 1
haz1_, haz3, Surv3 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[5] = 0
cov[6] = 1
haz1_, haz4, Surv4 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[6] = 0
cov[7] = 1
haz1_, haz5, Surv5 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[7] = 0
cov[8] = 1
haz1_, haz6, Surv6 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
cov[8] = 0
cov[9] = 1
haz1_, haz7, Surv7 = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
plot(t,haz1)
plot!(t,haz2)
plot!(t,haz3)
plot!(t,haz4)
plot!(t,haz5)
plot!(t,haz6)
plot!(t,haz7)

plot(t,Surv1, ylimits = (0,1))
plot!(t,Surv2)
plot!(t,Surv3)
plot!(t,Surv4)
plot!(t,Surv5)
plot!(t,Surv6)
plot!(t,Surv7)
savefig(plotsdir("TKT","Surv4.png"))

