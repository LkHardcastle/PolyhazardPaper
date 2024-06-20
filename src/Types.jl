abstract type Prior end

mutable struct PriorDefault <: Prior
    σ_α::Float64
    σ_β0::Float64
    σ_β::Float64
    K_prob::Vector{Float64}
    K_max::Int
    w_vs::Float64
    dists::Vector{Int64}
end

mutable struct PriorHyper <: Prior
    σ_α::Float64
    σ_β0::Float64
    σ_β::Float64
    K_prob::Vector{Float64}
    K_max::Int
    w_vs::Float64
    dists::Vector{Int64}
    a_vs::Float64
    b_vs::Float64
    ζ_1::Float64
    ζ_2::Float64
end


mutable struct PriorNoVS <: Prior
    σ_α::Float64
    σ_β0::Float64
    σ_β::Float64
    K_prob::Vector{Float64}
    K_max::Int
    dists::Vector{Int64}
end

struct PolyData
    # Length n vectors
    y::Vector{Float64}
    cens::Vector{Float64}
    # p x n matrix
    cov::Matrix{Float64}
    n::Int64
    p::Int64
end

mutable struct Dynamics
    ind::Int64
    skel::Int64
    u::Vector{Float64}
    u_dist::Int64
    h_death::Int64
    start_grad::Bool
    end_grad::Float64
    flip_j::Int64
    flip_k::Int64
    rj_move::Int64
    type::Int64
    a::Float64
    b::Float64
    flip_t::Float64
    h_swap::Int64
    swap_dist::Int64
    adapt_h::Float64
    adapt_μ::Vector{Float64}
    adapt_Σ::Matrix{Float64}
    u_all::Matrix{Float64}
    dist_all::Vector{Int64}
end

struct Settings
    max_ind::Int64
    max_skel::Int64
    t_stop::Float64
    t_lim::Float64
    grad_stop::Int64
    grad_lim::Int64
    rj_rate::Float64
    auto_add::Float64
    smp_rate::Float64
    swap::Float64
    exp_bound::Float64
    quant::Float64
    burn::Float64
    gibbs::Float64
    adapt_rate::Float64
    swap1_prob::Float64
    swap2_prob::Float64
end

mutable struct SamplerEval
    grad_bound::Int64
    grad_thin::Int64
    lhood_thin::Int64
    err_track::Int64
    births::Int64
    b_attempt::Int64
    deaths::Int64
    d_attempt::Int64
    swaps::Int64
    s_attempt::Int64
    v_updates::Int64
    v_attempt::Int64
    swaps1::Int64
    s_attempt1::Int64
    swaps2::Int64
    s_attempt2::Int64
end

mutable struct TMax
    event_times::Vector{Float64}
    t_max::Float64
end