function rj_attempt!(x::Array{Float64}, K::Int64, dist::Vector{Int64}, priors::Prior, dat::PolyData, dyn::Dynamics, settings::Settings)
    ## Pick move type 
    if rand() < settings.gibbs
        dyn.rj_move = 4
    elseif rand() > settings.swap
        if K == priors.K_max
            dyn.rj_move = 2
        elseif K == 1
            dyn.rj_move = 1
        else
            dyn.rj_move = rand(DiscreteUniform(1,2))
        end
    else
        if rand() < settings.swap1_prob
            dyn.rj_move = 3
        elseif rand() < settings.swap2_prob
            dyn.rj_move = 5
        else
            dyn.rj_move = 6
        end
    end
    if dyn.rj_move == 1
        rj_u!(dyn, priors)
        A = U_birth(x, K, dist, priors, dat, dyn)
        if isnan(A)
            println(x)
            println(dist);println(dyn.u);println(dyn.u_dist)
            error("Numerically unstable birth")
        end
    end
    if dyn.rj_move == 2
        dyn.h_death = rand(DiscreteUniform(1,K))
        A = U_death(x, K, dist, priors, dat, dyn)
        if isnan(A)
            error("Numerically unstable death")
        end
    end
    if dyn.rj_move == 3
        rj_swap!(dyn, priors, K, dist)
        A = U_swap(x, K, dist, priors, dat, dyn)
        if isnan(A)
            error("Numerically unstable swap")
        end
    end
    if dyn.rj_move == 5
        rj_med_match!(x, dat, dist, dyn, priors, K)
        A = U_swap(x, K, dist, priors, dat, dyn)
        if dyn.swap_dist == 1
            # LL -> W 
            A += x[1,dyn.h_swap]
        else
            # W -> LL 
            A -= x[1,dyn.h_swap]
        end
    end
    #if dyn.rj_move == 5
    #    rj_swap_new!(x, dyn, priors, K, dist)
    #    A = U_swap(x, K, dist, priors, dat, dyn)
    #    if isnan(A)
    #        error("Numerically unstable swap")
    #    end
    #end
    #if dyn.rj_move == 6
    #    rj_swap_all!(x, dat, dyn, priors, K)
    #    A = U_swap_all(x, K, dist, priors, dat, dyn)
    #    if isnan(A)
    #        error("Numerically unstable swap")
    #    end
    #end
    if dyn.rj_move == 4
        return 1
    end
    if rand() < min(1,exp(A))
        ## Accept/reject
        return 1
    else 
        return 0
    end
end

function rj_u!(dyn::Dynamics, priors::Prior)
    dyn.u = vcat(rand(Normal(0,priors.σ_α)),rand(Normal(0,priors.σ_β0)), rand(Normal(0,priors.σ_β),dat.p - 1).*rand(Bernoulli(1-priors.w_vs),dat.p - 1) )
    dyn.u_dist = priors.dists[rand(DiscreteUniform(1,size(priors.dists,1)))]
end

function rj_swap!(dyn::Dynamics, priors::Prior, K::Int, dist::Vector{Int64})
    dyn.u = vcat(rand(Normal(0,priors.σ_α)),rand(Normal(0,priors.σ_β0)), rand(Normal(0,priors.σ_β),dat.p - 1).*rand(Bernoulli(1-priors.w_vs),dat.p - 1))
    dyn.h_swap = rand(DiscreteUniform(1,K))
    dyn.swap_dist = priors.dists[findall(priors.dists .!= dist[dyn.h_swap])][rand(DiscreteUniform(1,size(priors.dists,1)-1))]
end

function rj_swap_new!(x::Matrix{Float64}, dyn::Dynamics, priors::Prior, K::Int, dist::Vector{Int64})
    dyn.h_swap = rand(DiscreteUniform(1,K))
    dyn.u = vcat(rand(Normal(0,priors.σ_α)),rand(Normal(0,priors.σ_β0)), -x[3:end,dyn.h_swap])
    dyn.swap_dist = priors.dists[findall(priors.dists .!= dist[dyn.h_swap])][rand(DiscreteUniform(1,size(priors.dists,1)-1))]
end

function rj_swap_all!(x::Matrix{Float64}, dat::PolyData, dyn::Dynamics, priors::Prior, K::Int)
    dyn.dist_all = priors.dists[rand(DiscreteUniform(1,size(priors.dists,1)),K)]
    dyn.u_all = zeros(size(x))
    for k in 1:K
        dyn.u_all[:,k] = vcat(rand(Normal(0,priors.σ_α)),rand(Normal(0,priors.σ_β0)), rand(Normal(0,priors.σ_β),dat.p - 1).*rand(Bernoulli(1-priors.w_vs),dat.p - 1))
    end
end

function rj_med_match!(x::Matrix{Float64}, dat::PolyData, dist::Vector{Int64}, dyn::Dynamics, priors::Prior, K::Int)
    dyn.h_swap = rand(DiscreteUniform(1,K))
    dyn.swap_dist = priors.dists[findall(priors.dists .!= dist[dyn.h_swap])][rand(DiscreteUniform(1,size(priors.dists,1)-1))]
    if dyn.swap_dist == 1
        # LL -> W
        med_new = -exp(x[1,dyn.h_swap])*x[2,dyn.h_swap] + log(log(2))
    else
        # W -> LL
        med_new = (-x[2,dyn.h_swap] + log(log(2)))*exp(-x[1,dyn.h_swap])
    end
    dyn.u = vcat(x[1,dyn.h_swap], med_new, -x[3:end,dyn.h_swap])
end


function rj_update(x::Array{Float64}, v::Array{Float64}, dist::Vector{Int64}, K::Int64, VS_times::Array{Float64}, priors::PriorDefault, dyn::Dynamics, sampler_eval::SamplerEval)
    if dyn.rj_move == 1
        x, v, K, dist, VS_times = birth_update(x, v, dist, K, VS_times, dyn)
        sampler_eval.births += 1
    end
    if dyn.rj_move == 2
        x, v, K, dist, VS_times = death_update(x, v, dist, K, VS_times, dyn)
        sampler_eval.deaths += 1
    end
    if dyn.rj_move == 3
        x, v, K, dist, VS_times = swap_update(x, v, dist, K, VS_times, dyn)
        sampler_eval.swaps += 1
    end
    if dyn.rj_move == 4

    end
    return x, v, K, dist, VS_times
end

function rj_update(x::Array{Float64}, v::Array{Float64}, dist::Vector{Int64}, K::Int64, VS_times::Array{Float64}, priors::PriorHyper, dyn::Dynamics, sampler_eval::SamplerEval, settings::Settings)
    if dyn.rj_move == 1
        x, v, K, dist, VS_times = birth_update(x, v, dist, K, VS_times, dyn)
        sampler_eval.births += 1
    end
    if dyn.rj_move == 2
        x, v, K, dist, VS_times = death_update(x, v, dist, K, VS_times, dyn)
        sampler_eval.deaths += 1
    end
    if dyn.rj_move == 3  || dyn.rj_move == 5
        x, v, K, dist, VS_times = swap_update(x, v, dist, K, VS_times, dyn)
        if dyn.rj_move == 3 
            sampler_eval.swaps += 1
        else
            sampler_eval.swaps1 += 1
        end
    end
    if dyn.rj_move == 4
        priors, VS_times = gibbs_update(v, K, priors, VS_times)
        priors = variance_update!(x, v, priors, sampler_eval, dyn, settings)
    end
    if dyn.rj_move == 6
        x, v, K, dist, VS_times = swap_all_update(x, v, dist, K, VS_times, dyn)
        sampler_eval.swaps2 += 1
    end
    return x, v, K, dist, VS_times
end


function birth_update(x::Array{Float64}, v::Array{Float64}, dist::Vector{Int64}, K::Int64, VS_times::Array{Float64}, dyn::Dynamics)
    # Add column to x
    x = [x dyn.u]
    # Add dist to dist
    push!(dist, dyn.u_dist)
    # Add velocities
    v = [v (rand(Bernoulli(0.5), size(v,1)).*2 .- 1.0).*(dyn.u .!= 0.0)]
    # Update K
    K += 1
    VS_times = [VS_times  zeros(size(x,1))]
    # Bad... I know
    for j in 1:size(VS_times,1)
        if j < 3
            VS_times[j,K] = Inf
        elseif v[j,K] == 0.0
            VS_times[j,K] = rand(Exponential(priors.w_vs/(1-priors.w_vs)))
        elseif x[j,K]*v[j,K] < 0.0
            VS_times[j,K] = abs(x[j,K])
        else
            VS_times[j,K] = Inf
        end
    end
    return x, v, K, dist, VS_times
end

function death_update(x::Array{Float64}, v::Array{Float64}, dist::Vector{Int64}, K::Int64, VS_times::Array{Float64}, dyn::Dynamics)
    # Remove column from x
    x = x[:, 1:end .!= dyn.h_death]
    # Remove dist
    deleteat!(dist, dyn.h_death)
    # Remove velocities
    v = v[:, 1:end .!= dyn.h_death]
    VS_times = VS_times[:, 1:end .!= dyn.h_death]
    # Update K 
    K -= 1
    return x, v, K, dist, VS_times
end



function swap_update(x::Array{Float64}, v::Array{Float64}, dist::Vector{Int64}, K::Int64, VS_times::Array{Float64}, dyn::Dynamics)
    x = x[:, 1:end .!= dyn.h_swap]
    # Remove dist
    deleteat!(dist, dyn.h_swap)
    # Remove velocities
    v = v[:, 1:end .!= dyn.h_swap]
    VS_times = VS_times[:, 1:end .!= dyn.h_swap]
    x = [x dyn.u]
    # Add dist to dist
    push!(dist, dyn.swap_dist)
    # Add velocities
    v = [v (rand(Bernoulli(0.5), size(v,1)).*2 .- 1.0).*(dyn.u .!= 0.0)]
    VS_times = [VS_times  zeros(size(x,1))]
    # Bad... I know
    for j in 1:size(VS_times,1)
        if j < 3
            VS_times[j,K] = Inf
        elseif v[j,K] == 0.0
            VS_times[j,K] = rand(Exponential(priors.w_vs/(1-priors.w_vs)))
        elseif x[j,K]*v[j,K] < 0.0
            VS_times[j,K] = abs(x[j,K])
        else
            VS_times[j,K] = Inf
        end
    end
    return x, v, K, dist, VS_times
end

function swap_all_update(x::Array{Float64}, v::Array{Float64}, dist::Vector{Int64}, K::Int64, VS_times::Array{Float64}, dyn::Dynamics)
    x = copy(dyn.u_all)
    dist = copy(dyn.dist_all)
    v = (rand(Bernoulli(0.5), size(x)).*2 .- 1.0).*(x .!= 0.0)
    for j in 1:size(VS_times,1)
        for k in 1:K
            if j < 3
                VS_times[j,k] = Inf
            elseif v[j,k] == 0.0
                VS_times[j,k] = rand(Exponential(priors.w_vs/(1-priors.w_vs)))
            elseif x[j,k]*v[j,k] < 0.0
                VS_times[j,k] = abs(x[j,K])
            else
                VS_times[j,k] = Inf
            end
        end
    end
    return x, v, K, dist, VS_times
end

function gibbs_update(v::Matrix{Float64}, K::Int64, priors::PriorHyper, VS_times::Array{Float64})
    v_total = (dat.p - 1)*K
    v_exc = sum(v[3:end, :] .== 0.0)
    priors.w_vs = rand(Beta(priors.a_vs + v_exc, priors.b_vs + v_total - v_exc))
    VS_times[findall(v .== 0.0)] .= rand(Exponential(priors.w_vs/(1-priors.w_vs)),v_exc)
    return priors, VS_times
end

function variance_update!(x::Matrix{Float64}, v::Matrix{Float64}, priors::PriorHyper, sampler_eval::SamplerEval, dyn::Dynamics, settings::Settings)
    sampler_eval.v_attempt += 1
    β = x[3:end,:][findall(v[3:end,:] .!= 0.0)]
    # Half-normal scaled by factor 2 - cancels in Metropolis ratio
    lpdf_curr = sum(logpdf.(Normal(0,priors.σ_β), β)) + 
                logpdf(Normal(0,1), exp(priors.ζ_1)) + 
                logpdf(InverseGamma(0.5,0.5), exp(priors.ζ_2))
    u = rand(MvNormal([0.0,0.0], dyn.adapt_h.*dyn.adapt_Σ))
    σ_prop = exp(priors.ζ_1 + u[1]).*sqrt(exp(priors.ζ_2 + u[2]))
    lpdf_prop = sum(logpdf.(Normal(0,σ_prop), β)) + 
                logpdf(Normal(0,1), exp(priors.ζ_1 + u[1])) + 
                logpdf(InverseGamma(0.5,0.5), exp(priors.ζ_2 + u[2]))
    A = min(1, exp(lpdf_prop - lpdf_curr))
    if A > rand()
        priors.ζ_1 = priors.ζ_1 + u[1]
        priors.ζ_2 = priors.ζ_2 + u[2]
        priors.σ_β = σ_prop
        sampler_eval.v_updates += 1
    end
    if sampler_eval.v_attempt < 1_000
        lr = settings.adapt_rate^(sampler_eval.v_attempt + 1)
        dyn.adapt_h = exp(log(dyn.adapt_h) + lr*(A - 0.352))
        μ_old = copy(dyn.adapt_μ)
        x_μ = [priors.ζ_1,priors.ζ_2] .- μ_old
        dyn.adapt_μ .= dyn.adapt_μ .+ lr.*x_μ
        dyn.adapt_Σ .= dyn.adapt_Σ .+ lr.*(x_μ*transpose(x_μ) .- dyn.adapt_Σ)
        #println(dyn.adapt_h);println(dyn.adapt_Σ)
    end
    return priors
end