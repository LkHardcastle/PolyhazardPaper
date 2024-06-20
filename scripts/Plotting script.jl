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

### Hazard curves
t = 0.01:0.01:20
WW = plot_Wei_h([log(0.54),-1.12],t) .+ plot_Wei_h([log(2.54),-8.67+2.42],t)
plot(t,WW)
LL = plot_LL_h([0.4, log(exp(0.7)^(-1/exp(0.4)))],t) .+ plot_LL_h([1.5,log(exp(-10)^(-1/exp(1.5)))],t)
plot!(t,LL)
LLW = plot_LL_h([0.4, log(exp(0.7)^(-1/exp(0.4)))],t) .+ plot_LL_h([1.5,log(exp(-8)^(-1/exp(1.5)))],t) .+ plot_Wei_h([log(2.8),-7],t)
plot!(t,LLW)
LW =  plot_LL_h([log(2.7),log(exp(-0.2)^(-1/2.7))],t) .+ plot_Wei_h([log(2.7),-8.67+2.42],t)
plot!(t,LW)
plot_df = DataFrame(Time = collect(t), WW = WW, LL = LL, LLW = LLW, LW = LW)

R"""
p <- $plot_df %>%
    pivot_longer(WW:LW) %>%
    mutate(name = case_when(
        name == "WW" ~ "W-W",
        name == "LL" ~ "LL-LL",
        name == "LLW" ~ "LL-LL-W",
        name == "LW" ~ "LL-W"
    )) %>%
    ggplot(aes(x = Time, y = value, colour = name)) + geom_line() +
    theme_classic() + xlab("Time (arbitrary units)") + ylab("h(t)") +
    scale_colour_manual(values = cbPalette[c(2,4,6:7)]) +
    theme(legend.position = "bottom") +
    guides(col = guide_legend(title = ""))
ggsave($plotsdir("Paper","PolyH.pdf"),p,  height = 4, width = 6)
"""

#### Sampler trajectories

ZZS = DataFrame(CSV.File(datadir("sims","ExampleSkel.csv"), header=0))[1:150,:]
ZZSS = DataFrame(CSV.File(datadir("sims","ExampleSkel_Sticky.csv"), header=0))[1:500,:]
R"""
p1 <- $ZZS %>%
    ggplot(aes(x = Column4, y = Column5)) + geom_path(col = cbPalette[6]) +
    theme_classic() + xlab("") + ylab("")
p2 <- $ZZSS %>%
    ggplot(aes(x = Column3, y = Column7)) + geom_path(col = cbPalette[6]) +
    theme_classic() + xlab("Sampler time") + ylab("")
p3 <- plot_grid(p1,p2)
ggsave($plotsdir("Paper","ZZS.pdf"),p3,  height = 3, width = 6)
"""

#### Swap experiment 

out11 = JLD2.load(datadir("sims","SwapSim","out11.jld2"))
out12 = JLD2.load(datadir("sims","SwapSim","out12.jld2"))
out21 = JLD2.load(datadir("sims","SwapSim","out21.jld2"))
out22 = JLD2.load(datadir("sims","SwapSim","out22.jld2"))
out31 = JLD2.load(datadir("sims","SwapSim","out31.jld2"))
out32 = JLD2.load(datadir("sims","SwapSim","out32.jld2"))

out_p1 = posterior_probs_smps(out11, [1,2])
out_p2 = posterior_probs_smps(out21, [1,2])
out_p3 = posterior_probs_smps(out31, [1,2])

out11["Eval"].births/out11["Eval"].b_attempt
out11["Eval"].deaths/out11["Eval"].d_attempt
out21["Eval"].swaps/out21["Eval"].s_attempt
out31["Eval"].swaps1/out31["Eval"].s_attempt1
ind = union(findall(out_p1["Dist_probs"] .> 0.05),findall(out_p2["Dist_probs"] .> 0.05),findall(out_p3["Dist_probs"] .> 0.05))

smp_end = 10_000
d11 = transpose(dist_roll(out11, [1,2]))[2:smp_end,ind]
d12 = transpose(dist_roll(out12, [1,2]))[2:smp_end,ind]
d21 = transpose(dist_roll(out21, [1,2]))[2:smp_end,ind]
d22 = transpose(dist_roll(out22, [1,2]))[2:smp_end,ind]
d31 = transpose(dist_roll(out31, [1,2]))[2:smp_end,ind]
d32 = transpose(dist_roll(out32, [1,2]))[2:smp_end,ind]

Sampler = vcat(fill(1,smp_end*2 - 2), fill(2,smp_end*2 - 2), fill(3,smp_end*2 - 2))
Iter = repeat([1,2], inner = smp_end - 1, outer = 3)
Sample = repeat(1:(smp_end-1), inner = 1, outer = 6)
plot_df = DataFrame(hcat(Sampler, Iter, Sample ,vcat(d11, d12, d21, d22, d31, d32)), ["Sampler", "Iter", "Sample", "Dist1", "Dist2", "Dist3", "Dist4", "Dist5"])

R"""
p <- $plot_df %>%
    pivot_longer(Dist1:Dist5) %>%
    mutate(name = case_when(
        name == "Dist1" ~ "W",
        name == "Dist2" ~ "LL",
        name == "Dist3" ~ "LL-W",
        name == "Dist4" ~ "W-W",
        name == "Dist5" ~ "LL-LL"
    )) %>%
    mutate(Sampler = ifelse(Sampler == 1, "Birth-death",
                            ifelse(Sampler == 2, "Independent Swap",
                            "Med. match Swap"))) %>%
    ggplot(aes(x = Sample, y = value, group = interaction(Iter,name), colour = name)) + geom_line() +
    theme_classic() +
    theme(legend.position = "bottom",
        axis.text.x = element_blank()) + 
    scale_x_continuous(breaks = c(0,5000,10000)) +
    scale_colour_manual(name = "Model", values = cbPalette[c(1:2,4,6:7)]) + ylab("Posterior probability") + xlab("Sampler time (arbitrary units)") +
    facet_wrap(~ Sampler, ncol = 3)
ggsave($plotsdir("Paper","SwapExp.pdf"),p,  height = 3, width = 6)
p <- $plot_df %>%
    pivot_longer(Dist1:Dist5) %>%
    mutate(name = case_when(
        name == "Dist1" ~ "W",
        name == "Dist2" ~ "LL",
        name == "Dist3" ~ "LL-W",
        name == "Dist4" ~ "W-W",
        name == "Dist5" ~ "LL-LL"
    )) %>%
    subset(Sampler != 2) %>%
    mutate(Sampler = ifelse(Sampler == 1, "Birth-death",
                            ifelse(Sampler == 2, "Independent Swap",
                            "Med. match Swap"))) %>%
    ggplot(aes(x = Sample, y = value, group = interaction(Iter,name), colour = name)) + geom_line() +
    theme_classic() +
    theme(legend.position = "bottom",
        axis.text.x = element_blank()) + 
    scale_x_continuous(breaks = c(0,5000,10000)) +
    scale_colour_manual(name = "Model", values = cbPalette[c(1:2,4,6:7)]) + ylab("Posterior probability") + xlab("Sampler time (arbitrary units)") +
    facet_wrap(~ Sampler, ncol = 2)
ggsave($plotsdir("Paper","SwapExp.png"),p,  height = 4, width = 5)
"""


#### Hazard curves from Lung transplant data

df = DataFrame(CSV.File(datadir("exp_raw","DigLung.csv")))
y = df[:,:time]
cens = convert(Array{Float64},df[:,:event])
covar = transpose([ones(size(y,1)) df[:,:SLT]])
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))
out1 = JLD2.load(datadir("sims","Dem_dig","Dem_out1.jld2"))
out_p1 = posterior_probs_smps(out1, [1,2])
t = 0.01:0.05:15
haz1_, haz1 = hazard_quant(sort_out(out1, [1,2], dat), [1,-0.5], t, 0.08)
haz2_, haz2 = hazard_quant(sort_out(out1, [1,2], dat), [1,0.5], t, 0.08)
Dem1 = 0.54*exp(-1.12).*t.^(0.54-1) .+ 2.54*exp(-8.67).*t.^(2.54-1)
Dem2 = 0.54*exp(-1.12).*t.^(0.54-1) .+ 2.54*exp(-8.67 + 2.42).*t.^(2.54-1)
cov = [1,-0.5]
subhaz1 = subhazard_quant(sort_out(out1, [1,2], dat), [1,1], cov, t)
sub1 = subhaz1[1][:,1,1] + subhaz1[1][:,1,2]
cov = [1,0.5]
subhaz2 = subhazard_quant(sort_out(out1, [1,2], dat), [1,1], cov, t)
sub2 = subhaz2[1][:,1,1] + subhaz2[1][:,1,2]

plot_df = DataFrame(t = collect(t), Dem1 = Dem1, Dem2 = Dem2, haz1 = vec(haz1), haz2 = vec(haz2), sub1 = sub1, sub2 = sub2)
R"""
rdat <- $plot_df
p <- rdat %>%
    pivot_longer(cols = Dem1:sub2) %>%
    mutate(type = ifelse(name == "Dem1" | name == "Dem2", "Original",
                    ifelse(name == "haz1" | name == "haz2", "Avg. model", "P-W submodel")),
           cov = ifelse(substr(name,nchar(name),nchar(name)) == "1", "DLT", "SLT")) %>%
    ggplot(aes(x = t, y = value, colour = cov, linetype = type)) + geom_line() +
    theme_classic() + ylab("h(t)") + xlab("Time (years)") +
    theme(legend.position = "bottom",
            legend.key.width = unit(0.4, "cm")) +
    scale_colour_manual(values = cbPalette[6:7], name = "Transplant") +
    scale_linetype_manual(values = c("solid","dotted","dashed"),name = "Model") 
ggsave($plotsdir("Paper","Lung1.pdf"),p,  height = 4, width = 6)
#ggsave($plotsdir("Paper","Lung1.png"),p,  height = 4, width = 10)
"""
### Demiris sensitivity experiment

files = readdir(datadir("sims","Lung_exp_3"))
summ = DataFrame(Iter = [], prior1 = [], DL = [], DLq1 = [], DLq2 = [], SL = [], SLq1 = [], SLq2 = [], diff = [], diffq1 = [], diffq2 = []) 
ind = 1
ms = Vector{Vector{Float64}}()
p1 = [25.0,10.0,5.0,2.0]
for i in eachindex(p1)
    println(ind)
    #for k in 1:2
        out = JLD2.load(datadir("sims","Lung_exp_3",files[ind]))
        cov = [1,-0.5]
        ms1, dead = mean_survival2(out,cov, 0.01:0.02:50, [1,1])
        cov = [1,0.5]
        ms2, dead = mean_survival2(out,cov, 0.01:0.02:50, [1,1])
        dif = ms1 .- ms2
        push!(ms,dif)
        ind += 1
    #end
end
ms_plot = ms[1:4]
min_ = minimum(size.(ms_plot,1))
m_list1 = ms_plot[1][1:min_]
m_list2 = ms_plot[2][1:min_]
m_list3 = ms_plot[3][1:min_]
m_list4 = ms_plot[4][1:min_]
m_dat = DataFrame(hcat(m_list1, m_list2, m_list3, m_list4), ["25","10","5","2"])
mean(m_list4)
R"""
p <- $m_dat %>%
    pivot_longer("25":"2") %>%
    ggplot(aes(y = value, x = factor(name, levels = c("2","5","10","25")),fill = factor(name, levels = c("2","5","10","25")))) + 
    geom_boxplot(aes(middle = mean(value))) +
    theme_classic() + scale_fill_manual(values = cbPalette[c(3,6:8)], name = "Prior sd") +
    theme(legend.position = "none") + ylab("Mean survival difference (years)") + xlab("Prior standard deviation") +
    geom_hline(aes(yintercept = 1.90), linetype = "dashed") + geom_hline(aes(yintercept = 3.83), linetype = "dotted")
ggsave($plotsdir("Paper","Lung_sens.pdf"),p,  height = 3, width = 6)
#ggsave($plotsdir("Paper","Lung_sens.png"),p,  height = 4, width = 6)
"""
t = 0.01:0.05:20
covar = [1,0.5]
out1 = JLD2.load(datadir("sims","Lung_exp_3","out1.jld2"))
out2 = JLD2.load(datadir("sims","Lung_exp_3","out2.jld2"))
out3 = JLD2.load(datadir("sims","Lung_exp_3","out3.jld2"))
out4 = JLD2.load(datadir("sims","Lung_exp_3","out4.jld2"))
subhaz1 = subhazard_quant(sort_out(out1, [1,2], dat), [1,1], covar, t)
sub1 = subhaz1[1][:,1,1] + subhaz1[1][:,1,2]
subhaz2 = subhazard_quant(sort_out(out2, [1,2], dat), [1,1], covar, t)
sub2 = subhaz2[1][:,1,1] + subhaz2[1][:,1,2]
subhaz3 = subhazard_quant(sort_out(out3, [1,2], dat), [1,1], covar, t)
sub3 = subhaz3[1][:,1,1] + subhaz3[1][:,1,2]
subhaz4 = subhazard_quant(sort_out(out4, [1,2], dat), [1,1], covar, t)
sub4 = subhaz4[1][:,1,1] + subhaz4[1][:,1,2]
subhaz5 = subhazard_quant(sort_out(out5, [1,2], dat), [1,1], covar, t)
sub5 = subhaz5[1][:,1,1] + subhaz5[1][:,1,2]
subhaz6 = subhazard_quant(sort_out(out6, [1,2], dat), [1,1], covar, t)
sub6 = subhaz6[1][:,1,1] + subhaz6[1][:,1,2]
plot(t,sub1)
plot(t,sub2)
plot!(t,sub3)
plot!(t,sub4)
plot!(t,sub5)
plot!(t,sub6)
t = 0.01:0.05:15
out1 = JLD2.load(datadir("sims","Lung_exp_3","out10.jld2"))
covar = [1,-0.5]
subhaz1 = subhazard_quant(sort_out(out1, [1,2], dat), [1,1], covar, t)
sub1 = subhaz1[1][:,1,1] + subhaz1[1][:,1,2]
covar = [1,0.5]
subhaz2 = subhazard_quant(sort_out(out1, [1,2], dat), [1,1], covar, t)
sub2 = subhaz2[1][:,1,1] + subhaz2[1][:,1,2]
#plot(t,sub1)
plot(t,sub2)

covar = [1,-0.5]
out1 = JLD2.load(datadir("sims","Lung_exp_3","out20.jld2"))
subhaz1 = subhazard_quant(sort_out(out1, [1,2], dat), [1,1], covar, t)
sub1 = subhaz1[1][:,1,1] + subhaz1[1][:,1,2]
covar = [1,0.5]
subhaz2 = subhazard_quant(sort_out(out1, [1,2], dat), [1,1], covar, t)
sub2 = subhaz2[1][:,1,1] + subhaz2[1][:,1,2]
plot(t,sub1)
plot!(t,sub2)

#####
Random.seed!(8475)
t = 0.01:0.05:1000
pred = prior_predictive(2,2,[1,1],t, 50_000)
mean(pred)
quantile(pred, 0.05)
quantile(pred, 0.95)
pred = prior_predictive(2,5,[2,2],t, 50_000)
mean(pred)
quantile(pred, 0.05)
quantile(pred, 0.95)
pred = prior_predictive(2,5,[1,1],t, 50_000)
mean(pred)
quantile(pred, 0.05)
quantile(pred, 0.95)
pred = prior_predictive(2,10,[1,1],t, 50_000)
mean(pred)
quantile(pred, 0.05)
quantile(pred, 0.95)
pred = prior_predictive(2,25,[1,1],t, 50_000)
mean(pred)
quantile(pred, 0.05)
quantile(pred, 0.95)
pred = prior_predictive(2,50,[1,1],t, 50_000)
mean(pred)
quantile(pred, 0.05)
quantile(pred, 0.95)
pred = prior_predictive(2,100,[1,1],t, 50_000)
mean(pred)
quantile(pred, 0.05)
quantile(pred, 0.95)


#### Hazard ratios from COST data
df = DataFrame(CSV.File(datadir("exp_pro","cost_c.csv")))
y = df[:,:time]
cens = convert(Array{Float64},df[:,:status])
covar = transpose([ones(size(y,1)) Matrix(df[:,2:14])])
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))

out1 = JLD2.load(datadir("sims","COST","COST_out1.jld2"))
out2 = JLD2.load(datadir("sims","COST","COST_Weibull.jld2"))
out3 = JLD2.load(datadir("sims","COST","COST_LL.jld2"))

t = 0.01:0.05:10
cov = vcat([1],zeros(13))
h_ratios1 = zeros(200,13)
h_ratios2 = zeros(200,13)
h_ratiosL1 = zeros(200,13)
h_ratiosL2 = zeros(200,13)
m1 = zeros(size(sort_out(out1, [1,2], dat)["Out"],2), 13)
m2 = zeros(size(sort_out(out1, [1,2], dat)["Out"],2), 13)
mdiff = zeros(size(sort_out(out1, [1,2], dat)["Out"],2), 13)
h2LL1 = zeros(200,2,13)
h2LL2 = zeros(200,2,13)

ind = [1,2,10,12]
for j in ind
    if j in [1,12,13]
        cov_ = copy(cov)
        cov_[j + 1] = 0.674
        dead, h_ratios1[:,j] = hazard_quant(sort_out(out1, [1,2], dat), cov_, t, 0.01)
        dead, h_ratiosL1[:,j] = hazard_quant(sort_out(out3, [1,2], dat), cov_, t, 0.01)
        h2LL1[:,:,j] = subhazard_quant(sort_out(out1, [1,2], dat), [2,2], cov_, t)[1][:,1,:]
        m1[:,j] = mean_survival2(out1, cov_, 0.01:0.05:30, [1,2])[1]
        cov_[j + 1] = -0.674
        dead, h_ratios2[:,j] = hazard_quant(sort_out(out1, [1,2], dat), cov_, t, 0.01)
        dead, h_ratiosL2[:,j] = hazard_quant(sort_out(out3, [1,2], dat), cov_, t, 0.01)
        h2LL2[:,:,j] = subhazard_quant(sort_out(out1, [1,2], dat), [2,2], cov_, t)[1][:,1,:]
        m2[:,j] = mean_survival2(out1, cov_, 0.01:0.05:30, [1,2])[1]
        mdiff[:,j] = m1[:,j] .- m2[:,j]
    else
        cov_ = copy(cov)
        cov_[j + 1] = 0.5
        dead, h_ratios1[:,j] = hazard_quant(sort_out(out1, [1,2], dat), cov_, t, 0.01)
        dead, h_ratiosL1[:,j] = hazard_quant(sort_out(out3, [1,2], dat), cov_, t, 0.01)
        h2LL1[:,:,j] = subhazard_quant(sort_out(out1, [1,2], dat), [2,2], cov_, t)[1][:,1,:]
        m1[:,j] = mean_survival2(out1, cov_, 0.01:0.05:30, [1,2])[1]
        cov_[j + 1] = -0.5
        dead, h_ratios2[:,j] = hazard_quant(sort_out(out1, [1,2], dat), cov_, t, 0.01)
        dead, h_ratiosL2[:,j] = hazard_quant(sort_out(out3, [1,2], dat), cov_, t, 0.01)
        h2LL2[:,:,j] = subhazard_quant(sort_out(out1, [1,2], dat), [2,2], cov_, t)[1][:,1,:]
        m2[:,j] = mean_survival2(out1, cov_, 0.01:0.05:30, [1,2])[1]
        mdiff[:,j] = m1[:,j] .- m2[:,j]
    end
end

h_ratios1 = h_ratios1[:,ind]
h_ratios2 = h_ratios2[:,ind]
h_ratiosL1 = h_ratiosL1[:,ind]
h_ratiosL2 = h_ratiosL2[:,ind]
m1 = m1[:,ind]
m2 = m2[:,ind]
mdiff = mdiff[:,ind]
mean(m1, dims = 1)
mean(m2, dims = 1)
mean(mdiff,dims = 1)

q1 = zeros(3,4)
q2 = zeros(3,4)
m_summ = vcat(mean(m1, dims = 1), mean(m2, dims = 1), mean(mdiff,dims = 1))
for j in 1:4
    for i in 1:3
        if i == 1
            q1[i,j] = quantile(m1[:,j], 0.05)
            q2[i,j] = quantile(m1[:,j], 0.95)
        elseif i == 2
            q1[i,j] = quantile(m2[:,j], 0.05)
            q2[i,j] = quantile(m2[:,j], 0.95)
        elseif i == 3
            q1[i,j] = quantile(mdiff[:,j], 0.05)
            q2[i,j] = quantile(mdiff[:,j], 0.95)
        end
    end
end
m_summ = trunc.(m_summ, digits = 2)
q1 = trunc.(q1, digits = 2)
q2 = trunc.(q2, digits = 2)


h2LL1 = h2LL1[:,:,ind]
h2LL2 = h2LL2[:,:,ind]
h2LL11 = h2LL1[:,1,:]
h2LL12 = h2LL1[:,2,:]
h2LL21 = h2LL2[:,1,:]
h2LL22 = h2LL2[:,2,:]

colnam =  ["Age", "Sex", "Atrial Fibrilation", "Stoke Score"]
plot_df = hcat(DataFrame(t = vcat(collect(t),collect(t),collect(t),collect(t),collect(t),collect(t))),DataFrame(vcat(h_ratios1, h_ratios2, h2LL11, h2LL21, h2LL12, h2LL22),colnam), 
                DataFrame(cov = vcat(fill("1",200), fill("2",200), fill("3",200), fill("4",200), fill("5",200), fill("6",200))))

m_summ
R"""
plot_dat = $plot_df
p <- plot_dat %>%
    pivot_longer(cols = "Age":"Stoke Score") %>%
    ggplot(aes(x = t, y = value, colour = cov, linetype = cov)) + geom_line() +
    theme_classic() + scale_colour_manual(values = c(cbPalette[c(6:7,6:7,6:7)], "black")) + 
    scale_linetype_manual(values = c("solid", "solid", "dashed", "dashed", "dashed", "dashed","solid")) +
    theme(legend.position = "none") +
    facet_wrap( ~ name) + 
    ylab("h(t)") + xlab("Time (years)") +
    geom_label(data = data.frame(t = 4.5, value = 0.9, name = c("Age", "Sex", "Atrial Fibrilation", "Stoke Score"), cov = c("7","7","7","7"), 
                label = c("U:  3.57 ( 2.75,  4.56)   \nL:  6.87 ( 5.41,  8.70)    \nD: -3.30 (-4.40, -2.41)",
                          "U:  4.20 ( 3.22,  5.42)   \nL:  5.91 ( 4.63,  7.48)    \nD: -1.70 (-2.61, -0.88)",
                          "U:  4.52 ( 3.30, 6.00)   \nL:  5.52 ( 4.42, 6.91)    \nD: -1.00 (-2.09, 0.00)",
                          "U: 5.79 (4.51, 7.42) \nL: 4.10 (3.20, 5.18) \nD: 1.68 (0.96, 2.56)")), 
                aes(label = label), size = 3, hjust = "left", label.size = NA)
ggsave($plotsdir("Paper","COST2.pdf"),p,  height = 4, width = 6)
"""

h_rat1 = h_ratios1./h_ratios2
h_ratL = h_ratiosL1./h_ratiosL2

h_ratW = vec(exp.(mean(sort_out(out2,[1],dat)["Out"][ind .+ 4,:], dims = 2).*[2*0.674, 1, 1, 2*0.674]))
h_ratW_ = transpose(h_ratW)
h_matW = zeros(0,4)
for i in 1:200
    h_matW = vcat(h_matW, h_ratW_)
end
colnam =  ["Age", "Sex", "Atrial Fibrilation", "Stoke Score"]
avg_df = DataFrame(h_rat1, colnam)
L_df = DataFrame(h_ratL, colnam)
W_df = DataFrame(h_matW, colnam)
com_df = vcat(avg_df, L_df, W_df)
labels = DataFrame(labels = vcat(fill("Avg.",200),fill("LL",200),fill("W",200)))
plot_df = DataFrame(t = vcat(collect(t), collect(t), collect(t)))
plot_df = hcat(plot_df, com_df, labels) 

R"""
plot_dat = $plot_df
W_HRs = $h_ratW
p <- plot_dat %>%
    pivot_longer(cols = "Age":"Stoke Score") %>%
    ggplot(aes(x = t, y = value, colour = labels)) + geom_line() +
    theme_classic() + scale_colour_manual(values = cbPalette[c(4,6:7)], name = "Model") + 
    geom_hline(aes(yintercept = 1), linetype = "dashed") + 
    facet_wrap( ~ name) + ylab("Hazard ratio") + xlab("Time (years)")
ggsave($plotsdir("Paper","COST1.pdf"),p,  height = 4, width = 6)
"""


# TKT data


out1 = JLD2.load(datadir("sims","TKT","TKT_out1.jld2"))
out2 = JLD2.load(datadir("sims","TKT","TKT_out2.jld2"))
t = 0.01:0.4:60
t_ = 0.01:0.4:100
df = DataFrame(CSV.File(datadir("exp_pro","KT_data.csv")))
y = df[:,:time]
cens = convert(Array{Float64},df[:,:event])
covar = transpose([ones(size(y,1)) df[:,:Hypertension] df[:,:sex] df[:,:Dyslipidemia] df[:,:age2] df[:,:age3] df[:,:age4] df[:,:age5] df[:,:age6] df[:,:age7]  df[:,:wait2] df[:,:wait3] df[:,:wait4]])
dat = PolyData(y, cens , covar, size(covar,2), size(covar,1))

sum(cens)/length(cens)
sum(cens[findall(df[:,:age2] .== 1)])/length(cens[findall(df[:,:age2] .== 1)])
sum(cens[findall(df[:,:age7] .== 1)])/length(cens[findall(df[:,:age7] .== 1)])

cov = vcat(1,zeros(3),fill(-0.5,6),fill(-0.5,3))
Surv = zeros(length(t),7,4)
m_s = zeros(size(out1["t"],1),7,4)
haz1_, haz1, Surv[:,1,1] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,1,1] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[5] = 0.5
haz1_, haz2, Surv[:,2,1] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,2,1] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[5] = -0.5
cov[6] = 0.5
haz1_, haz3, Surv[:,3,1] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,3,1] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[6] = -0.5
cov[7] = 0.5
haz1_, haz4, Surv[:,4,1] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,4,1] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[7] = -0.5
cov[8] = 0.5
haz1_, haz5, Surv[:,5,1] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,5,1] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[8] = -0.5
cov[9] = 0.5
haz1_, haz6, Surv[:,6,1] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,6,1] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[9] = -0.5
cov[10] = 0.5
haz1_, haz7, Surv[:,7,1] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,7,1] = mean_survival2(out1, cov, t_, [1,2])[1]

cov = vcat(1,zeros(3),fill(-0.5,6),0.5,-0.5,-0.5)
haz1_, haz1, Surv[:,1,2] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,1,2] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[5] = 0.5
haz1_, haz2, Surv[:,2,2] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,2,2] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[5] = -0.5
cov[6] = 0.5
haz1_, haz3, Surv[:,3,2] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,3,2] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[6] = -0.5
cov[7] = 0.5
haz1_, haz4, Surv[:,4,2] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,4,2] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[7] = -0.5
cov[8] = 0.5
haz1_, haz5, Surv[:,5,2] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,5,2] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[8] = -0.5
cov[9] = 0.5
haz1_, haz6, Surv[:,6,2] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,6,2] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[9] = -0.5
cov[10] = 0.5
haz1_, haz7, Surv[:,7,2] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,7,2] = mean_survival2(out1, cov, t_, [1,2])[1]

cov = vcat(1,zeros(3),fill(-0.5,6),-0.5,0.5,-0.5)
haz1_, haz1, Surv[:,1,3] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,1,3] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[5] = 0.5
haz1_, haz2, Surv[:,2,3] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,2,3] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[5] = -0.5
cov[6] = 0.5
haz1_, haz3, Surv[:,3,3] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,3,3] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[6] = -0.5
cov[7] = 0.5
haz1_, haz4, Surv[:,4,3] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,4,3] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[7] = -0.5
cov[8] = 0.5
haz1_, haz5, Surv[:,5,3] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,5,3] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[8] = -0.5
cov[9] = 0.5
haz1_, haz6, Surv[:,6,3] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,6,3] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[9] = -0.5
cov[10] = 0.5
haz1_, haz7, Surv[:,7,3] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,7,3] = mean_survival2(out1, cov, t_, [1,2])[1]

cov = vcat(1,zeros(3),fill(-0.5,6),-0.5,-0.5,0.5)
haz1_, haz1, Surv[:,1,4] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,1,4] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[5] = 0.5
haz1_, haz2, Surv[:,2,4] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,2,4] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[5] = -0.5
cov[6] = 0.5
haz1_, haz3, Surv[:,3,4] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,3,4] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[6] = -0.5
cov[7] = 0.5
haz1_, haz4, Surv[:,4,4] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,4,4] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[7] = -0.5
cov[8] = 0.5
haz1_, haz5, Surv[:,5,4] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,5,4] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[8] = -0.5
cov[9] = 0.5
haz1_, haz6, Surv[:,6,4] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,6,4] = mean_survival2(out1, cov, t_, [1,2])[1]
cov[9] = -0.5
cov[10] = 0.5
haz1_, haz7, Surv[:,7,4] = hazard_quant(sort_out(out1, [1,2], dat), cov, t, 0.01)
m_s[:,7,4] = mean_survival2(out1, cov, t_, [1,2])[1]

Surv
mean(m_s,dims = 1)

histogram(m_s[:,1:3,1])
histogram(m_s[:,4:7,1])
histogram(m_s[:,1:3,2])
histogram(m_s[:,4:7,2])
histogram(m_s[:,1:3,3])
histogram(m_s[:,4:7,3])
histogram(m_s[:,1:3,4])
histogram(m_s[:,4:7,4])

histogram(m_s[:,1,4:-1:1])
histogram(m_s[:,2,4:-1:1])
histogram(m_s[:,3,4:-1:1])
histogram(m_s[:,4,4:-1:1])
histogram(m_s[:,5,4:-1:1])
histogram(m_s[:,6,4:-1:1])
histogram(m_s[:,7,4:-1:1])

plot(t,Surv[:,1:7,1])
plot(t,Surv[:,1:7,2])
plot(t,Surv[:,1:7,3])
plot(t,Surv[:,1:7,4])

Surv2 = DataFrame(hcat(repeat(collect(t),4), repeat([1,2,3,4], inner = 150, outer = 1), vcat(Surv[:,:,1],Surv[:,:,2],Surv[:,:,3],Surv[:,:,4])), :auto)

R"""
r_dat = $Surv2
p <- r_dat %>%
    mutate(x2 = case_when(
        x2 == 1 ~ "<1 year",
        x2 == 2 ~ "1-3 years",
        x2 == 3 ~ "3-6 years",
        x2 == 4 ~ "6+ years"
    )) %>%
    pivot_longer(x3:x9) %>%
    ggplot(aes(x = x1, y = value, colour = name)) + geom_line() +
    theme_classic() + scale_colour_manual(values = cbPalette[c(1:4,6:9)], name = "Age group", labels = c("11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80")) +
    theme(legend.position = "right") + #geom_vline(xintercept = 18.85) + 
    facet_wrap(~ x2) + ylab("S(t)") + xlab("Time (years)") 
    ggsave($plotsdir("Paper","TKT1.pdf"),p,  height = 4, width = 6)
    ggsave($plotsdir("Paper","TKT1.png"),p,  height = 4, width = 8)
"""

m = mean(m_s,dims = 1)
m_diff = zeros(1,7,3)
q1 = zeros(1,7,4)
q2 = zeros(1,7,4)
q3 = zeros(1,7,4)
q4 = zeros(1,7,4)
q1d = zeros(1,7,3)
q2d = zeros(1,7,3)
q3d = zeros(1,7,3)
q4d = zeros(1,7,3)
for i in 1:7
    for j in 1:4
        q1[1,i,j] = quantile(m_s[:,i,j],0.05)
        q2[1,i,j] = quantile(m_s[:,i,j],0.25)
        q3[1,i,j] = quantile(m_s[:,i,j],0.75)
        q4[1,i,j] = quantile(m_s[:,i,j],0.95)
        if j < 4
            m_diff[1,i,j] = mean(m_s[:,i,j] .- m_s[:,i,j+1])
            q1d[1,i,j] = quantile(m_s[:,i,j] .- m_s[:,i,j+1],0.05)
            q2d[1,i,j] = quantile(m_s[:,i,j] .- m_s[:,i,j+1],0.25)
            q3d[1,i,j] = quantile(m_s[:,i,j] .- m_s[:,i,j+1],0.75)
            q4d[1,i,j] = quantile(m_s[:,i,j] .- m_s[:,i,j+1],0.95)
        end
    end
end
m
q1
q2


plot_df = DataFrame(hcat(vec(m_s[1:10:end,:,:]), repeat(collect(1:7), inner = size(m_s[1:10:end,:,:],1), outer = 4), repeat(collect(1:4), inner = 7, outer = size(m_s[1:10:end,:,:],1))), ["smp", "Age", "Wait"])
R"""
$plot_df %>% 
    ggplot(aes(x = as.factor(Wait), y = smp)) + geom_boxplot(aes(middle = mean(smp))) +
    theme_classic() + ylim(0,NA) +
    facet_wrap(~ as.factor(Age), ncol = 4)
"""

plot_df = DataFrame(hcat(vec(m), vec(q1), vec(q2), vec(q3), vec(q4), repeat(collect(1:7), inner = 1, outer = 4), repeat(collect(1:4), inner = 7, outer = 1)), ["Mean", "q1", "q2", "q3", "q4", "Age", "Wait"])
R"""
p <- $plot_df %>% 
    #pivot_longer(Mean:q2) %>%
    mutate(Age = case_when(
        Age == 1 ~ "11-20",
        Age == 2 ~ "21-30",
        Age == 3 ~ "31-40",
        Age == 4 ~ "41-50",
        Age == 5 ~ "51-60",
        Age == 6 ~ "61-70",
        Age == 7 ~ "71-80"
        ),
        Wait = case_when(
            Wait == 1 ~ "<1",
            Wait == 2 ~ "1-3",
            Wait == 3 ~ "3-6",
            Wait == 4 ~ "6+"
        )
    ) %>%
    ggplot(aes(x = Wait, y = Mean)) + geom_point(colour = cbPalette[6]) +
    geom_errorbar(aes(ymin = q1, ymax = q4), width = .2, colour = cbPalette[7] ) +
    geom_errorbar(aes(ymin = q2, ymax = q3), width = .5, colour = cbPalette[6]) +
    theme_classic() + ylim(0,NA) + xlab("Wait time (years)") + ylab("Mean survival (years)") + 
    facet_wrap(~ Age, ncol = 4)
    ggsave($plotsdir("Paper","TKT2.pdf"),p,  height = 4, width = 6)
    ggsave($plotsdir("Paper","TKT2.png"),p,  height = 4, width = 8)
"""

plot_df = DataFrame(hcat(vec(m_diff), vec(q1d), vec(q2d), vec(q3d), vec(q4d), repeat(collect(1:7), inner = 1, outer = 3), repeat(collect(1:3), inner = 7, outer = 1)), ["Mean", "q1", "q2", "q3", "q4", "Age", "Wait"])
R"""
p <- $plot_df %>% 
    #pivot_longer(Mean:q2) %>%
    mutate(Age = case_when(
        Age == 1 ~ "11-20",
        Age == 2 ~ "21-30",
        Age == 3 ~ "31-40",
        Age == 4 ~ "41-50",
        Age == 5 ~ "51-60",
        Age == 6 ~ "61-70",
        Age == 7 ~ "71-80"
        ),
        Wait = case_when(
            Wait == 1 ~ "1 vs 2",
            Wait == 2 ~ "2 vs 3",
            Wait == 3 ~ "3 vs 4",
        )
    ) %>%
    ggplot(aes(x = Wait, y = Mean)) + geom_point(colour = cbPalette[6]) +
    geom_errorbar(aes(ymin = q1, ymax = q4), width = .2, colour = cbPalette[7] ) +
    geom_errorbar(aes(ymin = q2, ymax = q3), width = .5, colour = cbPalette[6]) +
    theme_classic()  + xlab("Effect comparison") + ylab("Mean survival difference (years)") + geom_hline(aes(yintercept = 0), linetype = "dashed") +
    facet_wrap(~ Age, ncol = 4) #+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
    ggsave($plotsdir("Paper","TKT3.pdf"),p,  height = 4, width = 6)
    ggsave($plotsdir("Paper","TKT3.png"),p,  height = 4, width = 8)
"""