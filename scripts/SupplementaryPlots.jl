using DrWatson
@quickactivate "PolyhazardPaper"
using LinearAlgebra, Distributions, Optim, Random, Combinatorics
using DataFrames, RCall, Plots, CSV, DataFrames, JLD2, Combinatorics 

include(srcdir("Sampler.jl"))
include(srcdir("Postprocessing.jl"))

R"""
library(ggplot2)
library(dplyr)
library(tidyr)
library(cowplot)
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
"""
## Demiris trace plots

out1 = JLD2.load(datadir("sims","Dem","Dem_out1.jld2"))
out2 = JLD2.load(datadir("sims","Dem","Dem_out2.jld2"))
out3 = JLD2.load(datadir("sims","Dem","Dem_out3.jld2"))
df = DataFrame(CSV.File(datadir("exp_raw","DemirisLung.csv")))
y = df[:,:time]
cens = convert(Array{Float64},df[:,:event])
covar = transpose([ones(size(y,1)) df[:,:SLT] .- 0.5])
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))
# Dist DataFrame

out_p1 = posterior_probs_smps(out1, [1,2])
out_p2 = posterior_probs_smps(out2, [1,2])
ind = union(findall(out_p1["Dist_probs"] .> 0.01),findall(out_p2["Dist_probs"] .> 0.01))
smps1 = DataFrame(hcat(transpose(dist_roll(out1, [1,2]))[2:end,ind],fill(1,size(transpose(dist_roll(out1, [1,2]))[2:end,ind],1)), 1:size(transpose(dist_roll(out1, [1,2]))[2:end,ind],1)), :auto)
smps2 = DataFrame(hcat(transpose(dist_roll(out2, [1,2]))[2:end,ind],fill(2,size(transpose(dist_roll(out2, [1,2]))[2:end,ind],1)), 1:size(transpose(dist_roll(out2, [1,2]))[2:end,ind],1)), :auto)
R"""
$smps1 %>%
    pivot_longer(x1:x6) %>%
    ggplot(aes(x = x8, y = value, colour = name)) + geom_line() + 
    theme_classic() + theme(legend.position = "none") +
    scale_colour_manual(values = cbPalette[c(1:2,4,6:8)])
"""

R"""
$smps2 %>%
    pivot_longer(x1:x6) %>%
    ggplot(aes(x = x8, y = value, colour = name)) + geom_line() + 
    theme_classic() + theme(legend.position = "none") +
    scale_colour_manual(values = cbPalette[c(1:2,4,6:8)])
"""


sort1 = sort_out(out1, [1,2], dat)
plot(sort1["Out"][3,findall(sort1["Out"][2,:] .== 5)])
sort2 = sort_out(out2, [1,2], dat)
plot!(sort2["Out"][3,findall(sort2["Out"][2,:] .== 5)], legend = false)

t1 = sort1["Out"][3,findall(sort1["Out"][2,:] .== 5)]
t2 = sort2["Out"][3,findall(sort2["Out"][2,:] .== 5)][1:21330]

p1_df = DataFrame(t1 = Vector{Float64}(t1), t2 = Vector{Float64}(t2), Iter = collect(1:21330) .+ 0.0)

R"""
$p1_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
"""

t1 = sort1["Out"][3,findall(sort1["Out"][2,:] .== 9)]
t2 = sort2["Out"][3,findall(sort2["Out"][2,:] .== 9)][1:2397]

p2_df = DataFrame(t1 = Vector{Float64}(t1), t2 = Vector{Float64}(t2), Iter = collect(1:2397) .+ 0.0)

R"""
$p2_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
"""

R"""
p1 <- $smps1 %>%
    pivot_longer(x1:x6) %>%
    ggplot(aes(x = x8, y = value, colour = name)) + geom_line() + 
    theme_classic() + theme(legend.position = "none") +
    xlab("Iteration") + ylab("Posterior probability") +
    scale_colour_manual(values = cbPalette[c(1:2,4,6:8)])
p2 <- $smps2 %>%
    pivot_longer(x1:x6) %>%
    ggplot(aes(x = x8, y = value, colour = name)) + geom_line() + 
    theme_classic() + theme(legend.position = "none") +
    xlab("Iteration") + ylab("Posterior probability") +
    scale_colour_manual(values = cbPalette[c(1:2,4,6:8)])
p3 <- $p1_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
p4 <- $p2_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
p <- plot_grid(p1,p2,p3,p4, ncol = 2)
ggsave($plotsdir("Paper","Supp1.pdf"),p,  height = 4, width = 6)
"""

## COST trace plots

out1 = JLD2.load(datadir("sims","COST","COST_out1.jld2"))
out2 = JLD2.load(datadir("sims","COST","COST_out2.jld2"))
out3 = JLD2.load(datadir("sims","COST","COST_out3.jld2"))

df = DataFrame(CSV.File(datadir("exp_pro","cost_c.csv")))
y = df[:,:time]
cens = convert(Array{Float64},df[:,:status])
covar = transpose([ones(size(y,1)) Matrix(df[:,2:14])])
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))
# Dist DataFrame

out_p1 = posterior_probs_smps(out1, [1,2])
out_p2 = posterior_probs_smps(out2, [1,2])
ind = union(findall(out_p1["Dist_probs"] .> 0.01),findall(out_p2["Dist_probs"] .> 0.01))
smps1 = DataFrame(hcat(transpose(dist_roll(out1, [1,2]))[2:end,ind],fill(1,size(transpose(dist_roll(out1, [1,2]))[2:end,ind],1)), 1:size(transpose(dist_roll(out1, [1,2]))[2:end,ind],1)), :auto)
smps2 = DataFrame(hcat(transpose(dist_roll(out2, [1,2]))[2:end,ind],fill(2,size(transpose(dist_roll(out2, [1,2]))[2:end,ind],1)), 1:size(transpose(dist_roll(out2, [1,2]))[2:end,ind],1)), :auto)
R"""
$smps1 %>%
    pivot_longer(x1:x4) %>%
    ggplot(aes(x = x6, y = value, colour = name)) + geom_line() + 
    theme_classic() + theme(legend.position = "none") +
    scale_colour_manual(values = cbPalette[c(1:2,4,6:8)])
"""

R"""
$smps2 %>%
    pivot_longer(x1:x4) %>%
    ggplot(aes(x = x6, y = value, colour = name)) + geom_line() + 
    theme_classic() + theme(legend.position = "none") +
    scale_colour_manual(values = cbPalette[c(1:2,4,6:8)])
"""


sort1 = sort_out(out1, [1,2], dat)
plot(sort1["Out"][6,findall(sort1["Out"][2,:] .== 5)])
sort2 = sort_out(out2, [1,2], dat)
plot!(sort2["Out"][6,findall(sort2["Out"][2,:] .== 5)], legend = false)

t1 = sort1["Out"][6,findall(sort1["Out"][2,:] .== 5)][1:84730]
t2 = sort2["Out"][6,findall(sort2["Out"][2,:] .== 5)]

p1_df = DataFrame(t1 = Vector{Float64}(t1), t2 = Vector{Float64}(t2), Iter = collect(1:84730) .+ 0.0)

R"""
$p1_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
"""

sort1 = sort_out(out1, [1,2], dat)
plot(sort1["Out"][6,findall(sort1["Out"][2,:] .== 5)])
sort2 = sort_out(out2, [1,2], dat)
plot!(sort2["Out"][6,findall(sort2["Out"][2,:] .== 5)], legend = false)

t1 = sort1["Out"][7,findall(sort1["Out"][2,:] .== 5)][1:84730]
t2 = sort2["Out"][7,findall(sort2["Out"][2,:] .== 5)]

p2_df = DataFrame(t1 = Vector{Float64}(t1), t2 = Vector{Float64}(t2), Iter = collect(1:84730) .+ 0.0)

R"""
$p2_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
"""

t1 = out1["Hyper"]
t2 = out2["Hyper"]

p3_df = DataFrame(t1 = Vector{Float64}(transpose(t1)[1:97809,2]), t2 = Vector{Float64}(transpose(t2)[:,2]), Iter = collect(1:97809) .+ 0.0)

R"""
$p3_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
"""

p4_df = DataFrame(t1 = Vector{Float64}(transpose(t1)[1:97809,3]), t2 = Vector{Float64}(transpose(t2)[:,3]), Iter = collect(1:97809) .+ 0.0)

R"""
$p4_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
"""

R"""
p1 <- $smps1 %>%
    pivot_longer(x1:x4) %>%
    ggplot(aes(x = x6, y = value, colour = name)) + geom_line() + 
    theme_classic() + theme(legend.position = "none") +
    xlab("Iteration") + ylab("Posterior probability") +
    scale_colour_manual(values = cbPalette[c(1:2,4,6:8)])
p2 <- $smps2 %>%
    pivot_longer(x1:x4) %>%
    ggplot(aes(x = x6, y = value, colour = name)) + geom_line() + 
    theme_classic() + theme(legend.position = "none") +
    xlab("Iteration") + ylab("Posterior probability") +
    scale_colour_manual(values = cbPalette[c(1:2,4,6:8)])
p3 <- $p1_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
p4 <- $p2_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
p5 <- $p3_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
p6 <- $p4_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
p <- plot_grid(p1,p2,p3,p4,p5,p6, ncol = 2)
ggsave($plotsdir("Paper","Supp3.pdf"),p,  height = 6, width = 6)
"""

############### TKT supp plots

out1 = JLD2.load(datadir("sims","TKT","TKT_out1.jld2"))
out11 = JLD2.load(datadir("sims","TKT","TKT_out11.jld2"))
out12 = JLD2.load(datadir("sims","TKT","TKT_out12.jld2"))
out2 = JLD2.load(datadir("sims","TKT","TKT_out2.jld2"))
out21 = JLD2.load(datadir("sims","TKT","TKT_out21.jld2"))
out1_ = Dict([("Sk_x", hcat(out1["Sk_x"],out11["Sk_x"],out12["Sk_x"])), ("K", vcat(out1["K"],out11["K"],out12["K"])), ("t", vcat(out1["t"],out11["t"],out12["t"])), ("dist", hcat(out1["dist"],out11["dist"],out12["dist"])), ("Hyper", hcat(out1["Hyper"],out11["Hyper"],out12["Hyper"]))])
out2_ = Dict([("Sk_x", hcat(out2["Sk_x"],out21["Sk_x"])), ("K", vcat(out2["K"],out21["K"])), ("t", vcat(out2["t"],out21["t"])), ("dist", hcat(out2["dist"],out21["dist"])), ("Hyper", hcat(out2["Hyper"],out21["Hyper"]))])
out1_ = Dict([("Sk_x", hcat(out1["Sk_x"])), ("K", vcat(out1["K"])), ("t", vcat(out1["t"])), ("dist", hcat(out1["dist"])), ("Hyper", hcat(out1["Hyper"]))])
out2_ = Dict([("Sk_x", hcat(out2["Sk_x"])), ("K", vcat(out2["K"])), ("t", vcat(out2["t"])), ("dist", hcat(out2["dist"])), ("Hyper", hcat(out2["Hyper"]))])
df = DataFrame(CSV.File(datadir("exp_pro","KT_data.csv")))
y = df[:,:time]
cens = convert(Array{Float64},df[:,:event])
covar = transpose([ones(size(y,1)) df[:,:Hypertension] df[:,:sex] df[:,:Dyslipidemia] df[:,:age2] df[:,:age3] df[:,:age4] df[:,:age5] df[:,:age6] df[:,:age7]  df[:,:wait2] df[:,:wait3] df[:,:wait4]])
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))
sum(df[:,:Hypertension]) 
sum(df[:,:sex]) 
sum(df[:,:Dyslipidemia])
# Dist DataFrame
out1["Eval"]
out_p1 = posterior_probs_smps(out1, [1,2])
out_p2 = posterior_probs_smps(out2, [1,2])
ind = union(findall(out_p1["Dist_probs"] .> 0.05),findall(out_p2["Dist_probs"] .> 0.05))
smps1 = DataFrame(hcat(transpose(dist_roll(out1, [1,2]))[2:end,ind],fill(1,size(transpose(dist_roll(out1, [1,2]))[2:end,ind],1)), 1:size(transpose(dist_roll(out1, [1,2]))[2:end,ind],1)), :auto)
smps2 = DataFrame(hcat(transpose(dist_roll(out2, [1,2]))[2:end,ind],fill(2,size(transpose(dist_roll(out2, [1,2]))[2:end,ind],1)), 1:size(transpose(dist_roll(out2, [1,2]))[2:end,ind],1)), :auto)
R"""
$smps1 %>%
    pivot_longer(x1:x5) %>%
    ggplot(aes(x = x7, y = value, colour = name)) + geom_line() + 
    theme_classic() + theme(legend.position = "none") +
    scale_colour_manual(values = cbPalette[c(1:2,4,6:8)])
"""

R"""
$smps2 %>%
    pivot_longer(x1:x5) %>%
    ggplot(aes(x = x7, y = value, colour = name)) + geom_line() + 
    theme_classic() + theme(legend.position = "none") +
    scale_colour_manual(values = cbPalette[c(1:2,4,6:8)])
"""

covar = transpose([ones(size(y,1)) df[:,:Hypertension] df[:,:sex] df[:,:Dyslipidemia] df[:,:age2] df[:,:age3] df[:,:age4] df[:,:age5] df[:,:age6] df[:,:wait2] df[:,:wait3] df[:,:wait4]])
sort1 = sort_out(out1, [1,2], dat)
plot(sort1["Out"][11,findall(sort1["Out"][2,:] .== 4)])
sort2 = sort_out(out2, [1,2], dat)
plot!(sort2["Out"][6,findall(sort2["Out"][2,:] .== 4)], legend = false)

t1 = sort1["Out"][6,findall(sort1["Out"][2,:] .== 4)][1:80357]
t2 = sort2["Out"][6,findall(sort2["Out"][2,:] .== 4)][1:80357]

p1_df = DataFrame(t1 = Vector{Float64}(t1), t2 = Vector{Float64}(t2), Iter = collect(1:80357) .+ 0.0)

R"""
$p1_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
"""

sort1 = sort_out(out1, [1,2], dat)
plot(sort1["Out"][6,findall(sort1["Out"][2,:] .== 3)])
sort2 = sort_out(out2, [1,2], dat)
plot!(sort2["Out"][6,findall(sort2["Out"][2,:] .== 3)], legend = false)

t1 = sort1["Out"][3,findall(sort1["Out"][2,:] .== 3)][1:37736]
t2 = sort2["Out"][3,findall(sort2["Out"][2,:] .== 3)]

p2_df = DataFrame(t1 = Vector{Float64}(t1), t2 = Vector{Float64}(t2), Iter = collect(1:37736) .+ 0.0)

R"""
$p2_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
"""

t1 = out1["Hyper"]
t2 = out2["Hyper"]

p3_df = DataFrame(t1 = Vector{Float64}(transpose(t1)[1:191350,2]), t2 = Vector{Float64}(transpose(t2)[:,2]), Iter = collect(1:191350) .+ 0.0)

R"""
$p3_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
"""

p4_df = DataFrame(t1 = Vector{Float64}(transpose(t1)[1:191350,3]), t2 = Vector{Float64}(transpose(t2)[:,3]), Iter = collect(1:191350) .+ 0.0)

R"""
$p4_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
"""

R"""
p1 <- $smps1 %>%
    pivot_longer(x1:x5) %>%
    ggplot(aes(x = x7, y = value, colour = name)) + geom_line() + 
    theme_classic() + theme(legend.position = "none") +
    xlab("Iteration") + ylab("Posterior probability") +
    scale_colour_manual(values = cbPalette[c(1:2,4,6:8)])
p2 <- $smps2 %>%
    pivot_longer(x1:x5) %>%
    ggplot(aes(x = x7, y = value, colour = name)) + geom_line() + 
    theme_classic() + theme(legend.position = "none") +
    xlab("Iteration") + ylab("Posterior probability") +
    scale_colour_manual(values = cbPalette[c(1:2,4,6:8)])
p3 <- $p1_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
p4 <- $p2_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
p5 <- $p3_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
p6 <- $p4_df %>%
    pivot_longer(t1:t2) %>%
    ggplot(aes(x = Iter, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Iteration") + ylab("") + 
    theme(legend.position = "none") + 
    scale_colour_manual(values = cbPalette[6:7])
p <- plot_grid(p1,p2,p3,p4,p5,p6, ncol = 2)
ggsave($plotsdir("Paper","Supp3.pdf"),p,  height = 6, width = 6)
"""