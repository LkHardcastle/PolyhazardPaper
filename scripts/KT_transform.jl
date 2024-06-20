using DrWatson
@quickactivate "Polyhazard"
# For src
using LinearAlgebra, Distributions, Optim, Random, Combinatorics
# This script 
using Plots, CSV, DataFrames, JLD2, SurvivalAnalysis

include(srcdir("Sampler.jl"))
include(srcdir("Postprocessing.jl"))
include(srcdir("EDA.jl"))

dat = DataFrame(CSV.File(datadir("exp_raw","KT_data.csv")))

rename!(dat, :Waiting_time_for_KT => :Wait, :History_of_AMI => :AMI, :History_of_stroke => :stroke, :All_cause_death => :event, :All_cause_time => :time)
dat.sex = dat.sex .-1
dat.age_cat = (dat.age .== "11-20").*1 + (dat.age .== "21-30").*2 + (dat.age .== "31-40").*3 + (dat.age .== "41-50").*4 + 
                (dat.age .== "51-60").*5 + (dat.age .== "61-70").*6 + (dat.age .== "71-80").*7
dat.wait1 = (dat.Wait .== 0).*1.0
dat.wait2 = (dat.Wait .== 1).*1.0
dat.wait3 = (dat.Wait .== 2).*1.0
dat.wait4 = (dat.Wait .== 3).*1.0
dat.age2 = (dat.age .== "21-30").*1.0
dat.age3 = (dat.age .== "31-40").*1.0
dat.age4 = (dat.age .== "41-50").*1.0
dat.age5 = (dat.age .== "51-60").*1.0
dat.age6 = (dat.age .== "61-70").*1.0  
dat.age7 = (dat.age .== "71-80").*1.0
# Remove one patient with 0 survival time
dat = dat[setdiff(1:end,3401),:]

CSV.write(datadir("exp_pro","KT_data.csv"),dat)