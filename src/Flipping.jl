function flip_bound!(upper_bound::Float64, x::Array{Float64}, v::Array{Float64}, K::Int64, dist::Vector{Int64}, priors::Prior, dat::PolyData, dyn::Dynamics)
    ϵ = 0.01*upper_bound
    ##### Evaluations
    if !dyn.start_grad
        bounds = [globalrate(0.0, x, v, K, dist, priors, dat),
                globalrate(upper_bound/2, x, v, K, dist, priors, dat),
                globalrate(upper_bound, x, v, K, dist, priors, dat)]
        evals = 3
    else
        bounds = [dyn.end_grad,
            globalrate(upper_bound/2, x, v, K, dist, priors, dat),
            globalrate(upper_bound, x, v, K, dist, priors, dat)]
        evals = 2
    end
    dyn.end_grad = bounds[3]
    dyn.start_grad = true
    ##### Case when there is no rate
    if all(bounds .== 0.0)
        ### Return 0 as the bound
        Λbar = 0.0
        return Λbar, 0.0, evals
    end
    b_max = findmax(bounds)
    ##### Case when neither end point is a maximum
    if b_max[2] == 2
        ### Brent for constant bound
        optimΛlong  = optimize(t -> - globalrate(t, x, v, K, dist, priors, dat),0,upper_bound)
        Λbar = - Optim.minimum(optimΛlong)
        evals += Optim.f_calls(optimΛlong)
        return Λbar, 0.0, evals
    end
    ## Check if found maximum is a local maximum
    #if globalrate(ϵ*(b_max[2] == 1) + (upper_bound - ϵ)*(b_max[2] == 3), x, v, K, dat, y, cens, priors, dims) < b_max[1]
    #    evals += 1
        if bounds[2] < bounds[1] + ((bounds[3] -bounds[1])/upper_bound)*(upper_bound/2)
            ### Linear bound
            println(upper_bound);println(bounds)
            return bounds[1], ((bounds[3] -bounds[1])/upper_bound), evals
        else
            ### Constant bound
            Λbar = b_max[1]
            return Λbar, 0.0, evals
        end
    #end
    #### If endpoint isn't a local maximum then bound via numerical optimisation
    optimΛlong  = optimize(t -> - globalrate(t, x, v, K, dist, priors, dat),0,upper_bound)
    Λbar = - Optim.minimum(optimΛlong)
    evals += Optim.f_calls(optimΛlong)
    return Λbar, 0.0, evals
end

function flip_bound2!(upper_bound::Float64, x::Array{Float64}, v::Array{Float64}, K::Int64, dist::Vector{Int64}, priors::Prior, dat::PolyData, dyn::Dynamics, settings::Settings)
    ϵ = 0.01*upper_bound
    evals = 0
    ##### Evaluations
    if !dyn.start_grad
        bounds = [globalrate(0.0, x, v, K, dist, priors, dat),
                0.0,
                0.0]
        evals += 1
    else
        bounds = [dyn.end_grad,
            0.0,
            0.0]
    end
    #upper_bound = min(1/bounds[1],upper_bound)
    upper_bound = min(-log(1-settings.exp_bound)/bounds[1],upper_bound)
    bounds[2] = globalrate(upper_bound/2, x, v, K, dist, priors, dat)
    bounds[3] = globalrate(upper_bound, x, v, K, dist, priors, dat)
    evals += 2
    dyn.end_grad = bounds[3]
    dyn.start_grad = true
    ##### Case when there is no rate
    if all(bounds .== 0.0)
        ### Return 0 as the bound
        return 0.0, 0.0, evals, upper_bound
    end
    b_max = findmax(bounds)
    ##### Case when neither end point is a maximum
    if b_max[2] == 2
        ### Brent for constant bound
        bounds[3] = bounds[2]
        upper_bound = upper_bound/2
        bounds[2] = globalrate(upper_bound/2, x, v, K, dist, priors, dat)
        evals += 1
        b_max = findmax(bounds)
        if b_max[2] == 2
            if findmax(bounds)[1] > findmin(bounds)[1] + settings.auto_add/2
                optimΛlong  = optimize(t -> - globalrate(t, x, v, K, dist, priors, dat),0,upper_bound)
                Λbar = - Optim.minimum(optimΛlong)
                evals += Optim.f_calls(optimΛlong)
                return Λbar, 0.0, evals, upper_bound
            else
                Λbar = b_max[1]
                return Λbar, 0.0, evals, upper_bound
            end
        end
    end
    if b_max[2] == 3
        if bounds[2] < bounds[1] + ((bounds[3] -bounds[1])/2)
            ### Assume convex
            ## Check to see if should rebound
            if 0.5*(bounds[1] + 2*bounds[2] + bounds[3])/(bounds[1] + bounds[3]) < settings.quant
                bounds[3] = bounds[2]
                upper_bound = upper_bound/2
                bounds[2] = globalrate(upper_bound/2, x, v, K, dist, priors, dat)
                evals += 1
                if 0.5*(bounds[1] + 2*bounds[2] + bounds[3])/(bounds[1] + bounds[3]) < settings.quant
                    bounds[3] = bounds[2]
                    upper_bound = upper_bound/2
                    bounds[2] = globalrate(upper_bound/2, x, v, K, dist, priors, dat)
                    evals += 1
                end
            end
            #println(upper_bound);println(bounds)
            return bounds[1], ((bounds[3] -bounds[1])/upper_bound), evals, upper_bound
        else
            ### Assume concave - constant bound
            Λbar = b_max[1]
            return Λbar, 0.0, evals, upper_bound
        end
    end
    # Maximum at start - decend downhill (it's fine)
    if b_max[2] == 1
        if bounds[2] < bounds[1] + ((bounds[3] -bounds[1])/upper_bound)*(upper_bound/2)
            ### Linear bound
            #println(upper_bound);println(bounds)
            return bounds[1], ((bounds[3] -bounds[1])/upper_bound), evals, upper_bound
        else
            ### Constant bound
            Λbar = b_max[1]
            return Λbar, 0.0, evals, upper_bound
        end
    end
    #### If endpoint isn't a local maximum then bound via numerical optimisation
    optimΛlong  = optimize(t -> - globalrate(t, x, v, K, dist, priors, dat),0,upper_bound)
    Λbar = - Optim.minimum(optimΛlong)
    evals += Optim.f_calls(optimΛlong)
    return Λbar, 0.0, evals, upper_bound
end

function flip_attempt!(x::Array{Float64}, v::Array{Float64}, K::Int64, dist::Vector{Int64}, priors::Prior, dat::PolyData, dyn::Dynamics, sampler_eval::SamplerEval, t_int::Float64, settings::Settings)
    λ_i = λ_calc(x, v, K, dist, priors, dat)
    bound = dyn.a + t_int*dyn.b + settings.auto_add
    if sum(λ_i) > bound
        sampler_eval.err_track += 1
    end
    if rand() < sum(λ_i)/bound
        dyn.flip_k = rand(Categorical(sum(eachrow(λ_i))/sum(λ_i)))
        dyn.flip_j = rand(Categorical(λ_i[:,dyn.flip_k]/sum(λ_i[:,dyn.flip_k])))
        return 1
    else
        return 0
    end
end

function globalrate(t::Float64, x::Array{Float64}, v::Array{Float64}, K::Int64, dist::Vector{Int64}, priors::Prior, dat::PolyData)
    if t < 0.0
        error("t must be positive")
    end
    xt = x .+ v.*t
    Ux = ∇U(xt, K, dist, priors, dat)
    λi = max.(zeros(size(Ux)), Ux.*v)
    λt = sum(λi)
    return λt
end

function λ_calc(x::Array{Float64}, v::Array{Float64}, K::Int64, dist::Vector{Int64}, priors::Prior, dat::PolyData)
    Ux = ∇U(x, K, dist, priors, dat)
    λi = max.(zeros(size(Ux)), Ux.*v)
    return λi
end

function flip_update!(v::Array{Float64}, dyn::Dynamics)
    v[dyn.flip_j,dyn.flip_k] = -v[dyn.flip_j,dyn.flip_k]
end

function flip_update!(x::Array{Float64}, v::Array{Float64}, dyn::Dynamics, VS_times::Array{Float64})
    v[dyn.flip_j,dyn.flip_k] = -v[dyn.flip_j,dyn.flip_k]
    if dyn.flip_j > 2
        ## Update VS times
        if x[dyn.flip_j,dyn.flip_k]*v[dyn.flip_j,dyn.flip_k] < 0.0
            VS_times[dyn.flip_j,dyn.flip_k] = abs(x[dyn.flip_j,dyn.flip_k])/abs(v[dyn.flip_j,dyn.flip_k])
        else
            VS_times[dyn.flip_j,dyn.flip_k] = Inf
        end
    end
end