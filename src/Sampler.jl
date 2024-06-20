include(srcdir("Types.jl"))
include(srcdir("Potentials.jl"))
include(srcdir("Flipping.jl"))
include(srcdir("Reversible jump.jl"))
include(srcdir("VS.jl"))
function pdmp_sampler(x0::Array{Float64}, v0::Array{Float64}, t0::Float64, K0::Int64, dist0::Vector{Int64},
                        dat::PolyData, priors::PriorHyper, settings::Settings; sample = true)

    # Copy initial conditions
    x, v, t, K, dist = copy(x0), copy(v0), copy(t0), copy(K0), copy(dist0)
    ## Sampler evaluation
    sampler_eval = SamplerEval(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    dyn = Dynamics(1,0, [0.0], 0, 0,  false, 0.0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 1.68, [priors.ζ_1, priors.ζ_2], Diagonal(ones(2)),zeros(2,2),[])
    ## Adaptive t_max
    T_max = TMax(rand(Gamma(0.1,1),100),0.1)
    # Array with stick/unstick times
    # Point sampler downhill
    v .= -1 .* sign.(∇U(x, K, dist, priors, dat)).*abs.(v)
    VS_times = VS_time_start(x, v, dat, K)
    # Initialise Storage
    x_store = Array{Float64}(undef, priors.K_max*(dat.p + 1), settings.max_ind)
    x_store[1:(K*(dat.p + 1)),1] = vec(x)
    v_store = Array{Float64}(undef, priors.K_max*(dat.p + 1), settings.max_ind)
    v_store[1:(K*(dat.p + 1)),1] = vec(v)
    K_store = zeros(settings.max_ind)
    K_store[1] = copy(K) 
    t_store = zeros(settings.max_ind)
    t_store[1] = copy(t)
    hyper_store = Array{Float64}(undef, 3, settings.max_ind)
    hyper_store[1:3,1] = [priors.w_vs, priors.ζ_1, priors.ζ_2]
    dist_store = Array{Float64}(undef, priors.K_max, settings.max_ind)
    dist_store[1:K,1] = copy(dist)
    dist_poss = []
    dists_ = repeat(priors.dists,priors.K_max)
    for i in 1:trunc(Int,priors.K_max)
        perms = permutations(dists_, i)
        for j in perms
            push!(dist_poss, j)
        end
    end
    dist_poss = unique(sort.(dist_poss))
    dist_trans = zeros(length(dist_poss),length(dist_poss))

    dyn.ind += 1
    # Sampler loop
    println("Starting sampler")
    while dyn.skel < settings.max_skel
        ## PDMP inner
        
        t = pdmp_inner!(x, v, K, t, dist, priors, dat, dyn, VS_times, settings, T_max, sampler_eval)
        dist_old = copy(dist)
        if dyn.type ∈ [1,5] && !sample
            x_store[1:(K*(dat.p + 1)),dyn.ind] = vec(x)
            v_store[1:(K*(dat.p + 1)),dyn.ind] = vec(v)
            K_store[dyn.ind] = K
            t_store[dyn.ind] = t
            dist_store[1:K,dyn.ind] = dist
            hyper_store[1:3,dyn.ind] = [priors.w_vs, priors.ζ_1, priors.ζ_2]
            dyn.ind += 1
            dyn.skel += 1
        end
        # Update state of sampler
        x, v, K, dist, VS_times = pdmp_update(x, v, dist, K, VS_times, priors, dyn, sampler_eval)
        d_new = Inf
        if dyn.type == 5
            if dyn.rj_move != 4
                d_old = findfirst([sort(dist_old)] .== dist_poss)
                d_new = findfirst([sort(dist)] .== dist_poss)
                dist_trans[d_old,d_new] += 1
            end
        end
        if dyn.type == 4
            # Maybe put in function?
            popfirst!(T_max.event_times)
            push!(T_max.event_times, t - dyn.flip_t)
            T_max.t_max = sort(T_max.event_times)[95]
            dyn.flip_t = copy(t)
        end
        ## Storage
        if !sample || dyn.type == 3
            x_store[1:(K*(dat.p + 1)),dyn.ind] = copy(vec(x))
            v_store[1:(K*(dat.p + 1)),dyn.ind] = copy(vec(v))
            K_store[dyn.ind] = copy(K)
            t_store[dyn.ind] = copy(t)
            dist_store[1:K,dyn.ind] = copy(dist)
            hyper_store[1:3,dyn.ind] = [priors.w_vs, priors.ζ_1, priors.ζ_2]
            dyn.ind += 1
            if !sample
                dyn.skel += 1
            end
        end
        if sample
            dyn.skel += 1
        end
        # Print progress statement
        
        if mod(dyn.skel, settings.max_skel*0.1) == 0.0
            print("Iteration: ");print(dyn.skel);print("/");print(settings.max_skel);print("\n")
        end
        if end_checks(t, sampler_eval, settings,dyn)
            x_store = x_store[:,1:(dyn.ind-1)] 
            v_store = v_store[:,1:(dyn.ind-1)] 
            K_store = K_store[1:(dyn.ind-1)] 
            t_store = t_store[1:(dyn.ind-1)] 
            hyper_store = hyper_store[:,1:(dyn.ind-1)]
            dist_store = dist_store[:,1:(dyn.ind-1)]
            break
        end
    end
    brn = trunc(Int,dyn.ind*settings.burn) + 1
    x_store = x_store[:,brn:(dyn.ind-1)] 
    v_store = v_store[:,brn:(dyn.ind-1)] 
    K_store = K_store[brn:(dyn.ind-1)] 
    t_store = t_store[brn:(dyn.ind-1)]
    hyper_store = hyper_store[:,brn:(dyn.ind-1)]
    dist_store = dist_store[:,brn:(dyn.ind-1)]
    print("Iteration: ");print(dyn.ind);print("/");print(settings.max_ind);print("\n")
    print("Final time: ");print(t);print("\n")
    return Dict([("Sk_x", x_store), ("Sk_v", v_store), ("K", K_store), ("t", t_store), ("dist", dist_store), ("Hyper", hyper_store), ("Eval", sampler_eval), ("Trans",dist_trans)])
end

function pdmp_sampler(x0::Array{Float64}, v0::Array{Float64}, t0::Float64, K0::Int64, dist0::Vector{Int64},
                        dat::PolyData, priors::PriorDefault, settings::Settings; sample = true)

    # Copy initial conditions
    x, v, t, K, dist = copy(x0), copy(v0), copy(t0), copy(K0), copy(dist0)
    ## Sampler evaluation
    sampler_eval = SamplerEval(0,0,0,0,0,0,0,0,0,0)
    dyn = Dynamics(1,0, [0.0], 0, 0,  false, 0.0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0)
    ## Adaptive t_max
    T_max = TMax(rand(Gamma(0.1,1),100),0.1)
    # Array with stick/unstick times
    VS_times = VS_time_start(x, v, dat, K)
    # Initialise Storage
    x_store = Array{Float64}(undef, priors.K_max*(dat.p + 1), settings.max_ind)
    x_store[1:(K*(dat.p + 1)),1] = vec(x)
    v_store = Array{Float64}(undef, priors.K_max*(dat.p + 1), settings.max_ind)
    v_store[1:(K*(dat.p + 1)),1] = vec(v)
    K_store = zeros(settings.max_ind)
    K_store[1] = K 
    t_store = zeros(settings.max_ind)
    t_store[1] = t
    hyper_store = Array{Float64}(undef, 2, settings.max_ind)
    hyper_store[1,1] = priors.w_vs
    dist_store = Array{Float64}(undef, priors.K_max, settings.max_ind)
    dist_store[1:K,1] = dist
    dist_poss = []
    dists_ = repeat(priors.dists,priors.K_max)
    for i in 1:trunc(Int,priors.K_max)
        perms = permutations(dists_, i)
        for j in perms
            push!(dist_poss, j)
        end
    end
    dist_poss = unique(sort.(dist_poss))
    dist_trans = zeros(length(dist_poss),length(dist_poss))

    dyn.ind += 1
    # Sampler loop
    println("Starting sampler")
    while dyn.skel < settings.max_skel
        ## PDMP inner
        t = pdmp_inner!(x, v, K, t, dist, priors, dat, dyn, VS_times, settings, T_max, sampler_eval)
        #println(x);println(dyn.type);println(dyn.rj_move)
        dist_old = copy(dist)
        if dyn.type ∈ [1,5] && !sample
            x_store[1:(K*(dat.p + 1)),dyn.ind] = vec(x)
            v_store[1:(K*(dat.p + 1)),dyn.ind] = vec(v)
            K_store[dyn.ind] = K
            t_store[dyn.ind] = t
            dist_store[1:K,dyn.ind] = dist
            hyper_store[1,dyn.ind] = priors.w_vs
            dyn.ind += 1
        end
        # Update state of sampler
        x, v, K, dist, VS_times = pdmp_update(x, v, dist, K, VS_times, dyn, sampler_eval)
        d_new = Inf
        if dyn.type == 5
            d_old = findfirst([sort(dist_old)] .== dist_poss)
            d_new = findfirst([sort(dist)] .== dist_poss)
            #println(d_old);println(d_new);println(dist);println(dist)
            dist_trans[d_old,d_new] += 1
        end
        if dyn.type == 4
            # Maybe put in function?
            popfirst!(T_max.event_times)
            push!(T_max.event_times, t - dyn.flip_t)
            T_max.t_max = sort(T_max.event_times)[95]
            dyn.flip_t = t
        end
        #println("-------------")
        #println(dyn.type)
        #println(t);println(x);println(v);println(K);println(dist)
        ## Storage
        if !sample || dyn.type == 3
            x_store[1:(K*(dat.p + 1)),dyn.ind] = vec(x)
            v_store[1:(K*(dat.p + 1)),dyn.ind] = vec(v)
            K_store[dyn.ind] = K
            t_store[dyn.ind] = t
            dist_store[1:K,dyn.ind] = dist
            hyper_store[1,dyn.ind] = priors.w_vs
            dyn.ind += 1
        end
        if sample
            dyn.skel += 1
        end
        # Print progress statement
        if mod(dyn.skel, settings.max_skel*0.1) == 0.0
            print("Iteration: ");print(dyn.skel);print("/");print(settings.max_skel);print("\n")
        end
        if end_checks(t, sampler_eval, settings,dyn)
            x_store = x_store[:,1:(dyn.ind-1)] 
            v_store = v_store[:,1:(dyn.ind-1)] 
            K_store = K_store[1:(dyn.ind-1)] 
            t_store = t_store[1:(dyn.ind-1)] 
            hyper_store = hyper_store[:,(dyn.ind-1)]
            dist_store = dist_store[:,1:(dyn.ind-1)]
            break
        end
    end
    brn = trunc(Int,dyn.ind*settings.burn)
    x_store = x_store[:,brn:(dyn.ind-1)] 
    v_store = v_store[:,brn:(dyn.ind-1)] 
    K_store = K_store[brn:(dyn.ind-1)] 
    t_store = t_store[brn:(dyn.ind-1)]
    hyper_store = hyper_store[:,brn:(dyn.ind-1)]
    dist_store = dist_store[:,brn:(dyn.ind-1)]
    print("Iteration: ");print(dyn.ind);print("/");print(settings.max_ind);print("\n")
    print("Final time: ");print(t);print("\n")
    return Dict([("Sk_x", x_store), ("Sk_v", v_store), ("K", K_store), ("t", t_store), ("dist", dist_store), ("Hyper", hyper_store), ("Eval", sampler_eval), ("Trans",dist_trans)])
end

function end_checks(t::Float64, sampler_eval::SamplerEval, settings::Settings, dyn::Dynamics)
    if settings.t_stop < t && settings.grad_stop < (sampler_eval.grad_bound + sampler_eval.grad_thin)
        println("Stops reached")
        return true
    elseif settings.t_lim < t 
        println("Max time reached")
        return true
    elseif settings.grad_lim < (sampler_eval.grad_bound + sampler_eval.grad_thin)
        println("Max grad reached")
        return true
    elseif settings.max_ind < (dyn.ind)
        return true
    else
        return false
    end
end

function pdmp_inner!(x::Array{Float64}, v::Array{Float64}, K::Int64, t::Float64, dist::Vector{Int64}, priors::PriorHyper, dat::PolyData, dyn::Dynamics, 
                    VS_times::Array{Float64}, settings::Settings, T_max::TMax, sampler_eval::SamplerEval)
    ### Returns the next skeleton point
    inner_stop = false
    t_old = copy(t)
    re_bound = false
    dyn.type = 0
    while !inner_stop
        ## Finds the next set event time
        t_bound = find_times!(VS_times, dyn, max(T_max.t_max, 0.001), settings)
        if t_bound < 0.0
            println(x);println(v)
            println(VS_times)
            error("Badness")
        end
        if dyn.type == 1
            inner_stop = true
        end
        ## Generate flip bounds
        dyn.start_grad = false
        dyn.a, dyn.b, evals, t_bound_new = flip_bound2!(t_bound, x, v, K, dist, priors, dat, dyn, settings)
        if t_bound_new < t_bound
            if t_bound_new < 0.0
                println(x);println(v)
                println(VS_times)
                error("Badness")
            end
            t_bound = find_times!(VS_times, dyn, t_bound_new, settings)
            if dyn.type == 1
                inner_stop = true
            else 
                inner_stop = false
            end
        end
        sampler_eval.grad_bound += evals
        if abs(dyn.b) < 1e-10
            dyn.b = 0.0
        end
        if isnan(dyn.a)
            println("----------")
            println(dyn.a);println(dyn.b)
            println(x);println(v);println(dist)
            println(λ_calc(x, v, K, dist, priors, dat));
            println(K);println(priors)
            error("NaN rate")
        end
        if 10^5 < dyn.a
            println("Hello");println(t)
            #println(x)
            #println(λ_calc(x, v, K, dist, priors, dat))
            #println(dyn.a);
            #println(t_bound)
            hold = findmax(λ_calc(x, v, K, dist, priors, dat))[2]
            dyn.flip_j, dyn.flip_k = hold[1], hold[2]
            extra = 0.00001
            x .+= v.*extra
            t = t + extra
            VS_times .-= extra
            #println(x);println(VS_times)
            sampler_eval.grad_thin += 1
            inner_stop = true
            dyn.type = 4
            return t
        end
        t_int = 0.0
        its = 0
        while t_int < t_bound
            its += 1
            #println(its)
            if its > 1_000
                println("----------")
                println(t_bound)
                println(dyn.a);println(dyn.b)
                println(x);println(v);println(dist)
                println(λ_calc(x, v, K, dist, priors, dat));
                println(globalrate(0.0, x, v, K, dist, priors, dat))
                println(K)
                if !re_bound
                    println("Too long on single bound - re-bounding")
                    re_bound = true
                    dyn.start_grad = false
                    break
                else
                    println("Forcing")
                    hold = findmax(λ_calc(x, v, K, dist, priors, dat))[2]
                    dyn.flip_j, dyn.flip_k = hold[1], hold[2]
                    extra = 0.00001
                    x .+= v.*extra
                    t = t + extra
                    VS_times .-= extra
                    #println(x);println(VS_times)
                    sampler_eval.grad_thin += 1
                    inner_stop = true
                    dyn.type = 4
                    return t
                    #error("Too long on single bound - re-bounding")
                end
            end
            ## Generate new event time
            t_new = poisson_time(dyn.a + settings.rj_rate + settings.auto_add + settings.smp_rate, dyn.b, rand())
            if t_int + t_new > t_bound
                x .+= v.*(t_bound - t_int)
                t = t + (t_bound - t_int)
                VS_times .-= (t_bound - t_int)
                t_int = t_int + t_new
            else
                x .+= v.*t_new
                t = t + t_new
                VS_times .-= t_new
                t_int = t_int + t_new
                acc = pdmp_thin!(x, v, K, dist, priors, dat, dyn, settings, sampler_eval, t_int)
                if acc
                    inner_stop = true
                    return t
                end
            end
        end
    end
    return t
end

function pdmp_inner!(x::Array{Float64}, v::Array{Float64}, K::Int64, t::Float64, dist::Vector{Int64}, priors::PriorDefault, dat::PolyData, dyn::Dynamics, 
                    VS_times::Array{Float64}, settings::Settings, T_max::TMax, sampler_eval::SamplerEval)
    ### Returns the next skeleton point
    inner_stop = false
    t_old = copy(t)
    re_bound = false
    while !inner_stop
        ## Finds the next set event time
        t_bound = find_times!(VS_times, dyn, T_max, settings)
        if t_bound < 0.0
            println(x);println(v)
            println(VS_times)
        end
        if dyn.type == 1
            inner_stop = true
        end
        ## Generate flip bounds
        dyn.start_grad = false
        dyn.a, dyn.b, evals, t_bound_new = flip_bound2!(t_bound, x, v, K, dist, priors, dat, dyn, settings)
        if t_bound_new < t_bound
            t_bound = copy(t_bound_new)
            dyn.type = 2 
            inner_stop = false
        end
        sampler_eval.grad_bound += evals
        if isnan(dyn.a)
            println("----------")
            println(dyn.a);println(dyn.b)
            println(x);println(v);println(dist)
            println(λ_calc(x, v, K, dist, priors, dat));
            println(K)
            error("NaN rate")
        end
        if 10^5 < dyn.a
            hold = findmax(λ_calc(x, v, K, dist, priors, dat))[2]
            dyn.flip_j, dyn.flip_k = hold[1], hold[2]
            x += v.*0.00001
            t += 0.00001
            VS_times .-= 0.00001.*abs.(v)
            sampler_eval.grad_thin += 1
            inner_stop = true
            dyn.type = 4
            break
        end
        t_int = 0.0
        its = 0
        while t_int < t_bound
            its += 1
            #println(its)
            if its > 100
                println("----------")
                println(t_bound)
                println(dyn.a);println(dyn.b)
                println(x);println(v);println(dist)
                println(λ_calc(x, v, K, dist, priors, dat));
                println(globalrate(0.0, x, v, K, dist, priors, dat))
                println(K)
                if !re_bound
                    println("Too long on single bound - re-bounding")
                    dyn.start_grad = false
                    break
                else
                    error("Too long on single bound - re-bounding")
                end
            end
            ## Generate new event time
            t_new = poisson_time(dyn.a + settings.rj_rate + settings.auto_add + settings.smp_rate, dyn.b, rand())
            if t_int + t_new > t_bound
                x .+= v.*(t_bound - t_int)
                t += (t_bound - t_int)
                VS_times .-= (t_bound - t_int)
                t_int += t_new
            else
                x .+= v.*t_new
                t += t_new
                VS_times .-= t_new
                t_int += t_new
                inner_stop = pdmp_thin!(x, v, K, dist, priors, dat, dyn, settings, sampler_eval, t_int,inner_stop)
                if inner_stop
                    break
                end
            end
        end
    end
    return t
end

function poisson_time(a::Float64, b::Float64, u::Float64)
    ######## From ZigZagBoomerang.jl
    if b > 0
        if a < 0
            return sqrt(-log(u)*2.0/b) - a/b
        else # a[i]>0
            return sqrt((a/b)^2 - log(u)*2.0/b) - a/b
        end
    elseif b == 0
        if a > 0
            return -log(u)/a
        else # a[i] <= 0
            return Inf
        end
    else # b[i] < 0
        if a <= 0
            return Inf
        elseif -log(u) <= -a^2/b + a^2/(2*b)
            return -sqrt((a/b)^2 - log(u)*2.0/b) - a/b
        else
            return Inf
        end
    end
end

function find_times!(VS_times::Array{Float64}, dyn::Dynamics, t_max::Float64, settings::Settings)
    ## Finds the next event time
    t1 = findmin(VS_times)[1]
    ## Identify next non-flip event and 
    t_bound, dyn.type = findmin(vcat(t1, t_max))
    return t_bound
end

function pdmp_thin!(x::Array{Float64}, v::Array{Float64}, K::Int64, dist::Vector{Int64}, priors::Prior, dat::PolyData, dyn::Dynamics, settings::Settings, sampler_eval::SamplerEval, t_int::Float64)
    u = rand()
    Λ = (settings.rj_rate + dyn.a + settings.smp_rate + dyn.b*t_int + settings.auto_add)
    if u < (settings.rj_rate/Λ)
        # Attempt reversible jump
        acc = rj_attempt!(x, K, dist, priors, dat, dyn, settings)
        sampler_eval.lhood_thin += 1
        if dyn.rj_move == 1
            sampler_eval.b_attempt += 1
        end
        if dyn.rj_move == 2
            sampler_eval.d_attempt += 1
        end
        if dyn.rj_move == 3
            sampler_eval.s_attempt += 1
        end
        if dyn.rj_move == 5
            sampler_eval.s_attempt1 += 1
        end
        if dyn.rj_move == 6
            sampler_eval.s_attempt2 += 1
        end
        if acc == 1
            dyn.type = 5
            return true
        else
            return false
        end
    elseif u < ((settings.rj_rate + settings.smp_rate)/Λ)
        dyn.type = 3
        return true
    else
        # Attempt flip
        acc = flip_attempt!(x, v, K, dist, priors, dat, dyn, sampler_eval, t_int, settings)
        sampler_eval.grad_thin += 1
        if acc == 1
            dyn.type = 4
            return true
        else
            return false
        end
    end
end


function pdmp_update(x::Array{Float64}, v::Array{Float64}, dist::Vector{Int64}, K::Int64, VS_times::Matrix{Float64}, priors::Prior, dyn::Dynamics, sampler_eval::SamplerEval)
    dyn.start_grad = false
    if dyn.type == 1
        j = findmin(VS_times)[2]
        if v[j] != 0.0
            if x[j] != 0.0
                # Account for floating point errors
                if abs(x[j]) < 1e-16
                    x[j] = 0.0
                    VS_times[j] = 0.0
                else
                    println(dist)
                    println(abs(x[j]))
                    println(j);println(x);println(v);
                    println(x[j]);println(v[j]);
                    println(VS_times);
                    error("VS_times going wrong")
                end
            end
        end
        VS_update!(v, VS_times, priors)
    end
    if dyn.type == 4
        # Flip update
        flip_update!(x, v, dyn, VS_times)
    end
    if dyn.type == 5
        # RJ update
        x, v, K, dist, VS_times = rj_update(x, v, dist, K, VS_times, priors, dyn, sampler_eval, settings)
    end
    return x, v, K, dist, VS_times
end

