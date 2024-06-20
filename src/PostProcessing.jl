function posterior_probs(out::Dict, dists; VS_threshold = 0.1)
    # Takes the output of the pdmp sampler (i.e skeleton points)
    # from the polyhazard model and Returns
    # *K posterior probabilities
    # *dist posterior probabilites 
    # *VS posterior probabilites
    t = out["t"]
    K = out["K"]
    dist = out["dist"]
    x = transpose(out["Sk_x"])
    K_max = size(dist,1)
    K_probs = zeros(K_max)
    mod_probs = zeros(1)
    dist_poss = []
    for i in 1:trunc(Int,findmax(K)[1])
        perms = permutations(dists, i)
        for j in perms
            push!(dist_poss, j)
        end
    end
    dist_poss = unique(sort.(dist_poss))
    dist_probs = zeros(length(dist_poss))
    for i in eachindex(t)
        if i > 1
            K_probs[trunc(Int,K[i])] += (t[i] - t[i-1])
            if isnothing(findfirst([sort(dist[1:trunc(Int,K[i]),i])] .== dist_poss))
                println(i)
            end
            dist_probs[findfirst([dist[1:trunc(Int,K[i]),i]] .== dist_poss)] += (t[i] - t[i-1])
        end
    end
    K_probs = K_probs./sum(K_probs)
    dist_probs = dist_probs./sum(dist_probs)
    VS_probs = zeros(2,2)
    for i in eachindex(t)
        if i > 1
            if K[i] == 2 && K[i-1] == 2
                if x[i,3] == 0.0 && x[i-1,3] == 0.0
                    VS_probs[1,1] += (t[i] - t[i-1])
                else
                    VS_probs[1,2] += (t[i] - t[i-1])
                end
                if x[i,6] == 0.0 && x[i-1,6] == 0.0
                    VS_probs[2,1] += (t[i] - t[i-1])
                else
                    VS_probs[2,2] += (t[i] - t[i-1])
                end
            end
        end
    end
    VS_probs = VS_probs./sum(VS_probs[1,:])
    output = Dict([("mod_probs", mod_probs), ("K_probs", K_probs), ("VS_probs", VS_probs), ("Dist_probs", dist_probs), ("dist_poss", dist_poss)])
    return output
end

function posterior_probs_smps(out::Dict, dists; VS_threshold = 0.1)
    # Takes the output of the pdmp sampler (i.e skeleton points)
    # from the polyhazard model and Returns
    # *K posterior probabilities
    # *dist posterior probabilites 
    # *VS posterior probabilites
    K = out["K"]
    dist = out["dist"]
    x = transpose(out["Sk_x"])
    K_max = size(dist,1)
    K_probs = zeros(K_max)
    mod_probs = zeros(1)
    dist_poss = []
    dists = repeat(dists,trunc(Int,findmax(K)[1]))
    for i in 1:trunc(Int,findmax(K)[1])
        perms = permutations(dists, i)
        for j in perms
            push!(dist_poss, j)
        end
    end
    dist_poss = unique(sort.(dist_poss))
    dist_probs = zeros(length(dist_poss))
    VS_probs = zeros(size(dist_poss,1),trunc(Int,findmax(K)[1]),2)
    for i in eachindex(K)
        if i > 1
            K_curr = trunc(Int,K[i])
            K_probs[K_curr] += 1
            dist_curr = sort(dist[1:K_curr,i])
            dist_ind = findfirst([dist_curr] .== dist_poss)
            if isnothing(dist_ind)
                println(i)
            end
            dist_probs[dist_ind] += 1
            for k in 1:K_curr
                if x[i,(k-1)*3 + 3] == 0.0 
                    VS_probs[dist_ind,k,1] += 1
                else
                    VS_probs[dist_ind,k,2] += 1
                end
            end
        end
    end
    K_probs = K_probs./sum(K_probs)
    dist_probs = dist_probs./sum(dist_probs)
    VS_probs = VS_probs./sum(VS_probs, dims = 3)
    output = Dict([("mod_probs", mod_probs), ("K_probs", K_probs), ("VS_probs", VS_probs), ("Dist_probs", dist_probs), ("dist_poss", dist_poss)])
    return output
end

function K_roll(out)
    K = out["K"]
    K_probs = fill(0.0, trunc(Int,findmax(K)[1]),size(out["K"],1))
    for i in eachindex(K)
        if i >1
            K_probs[:,i] = K_probs[:,i-1]
            K_probs[trunc(Int,K[i]),i] += 1
        end
    end
    K_probs = K_probs./sum(K_probs, dims = 1)
    return K_probs
end

function dist_roll(out, dists)
    K = out["K"]
    dist_poss = []
    dist = out["dist"]
    dists = repeat(dists,trunc(Int,findmax(K)[1]))
    for i in 1:trunc(Int,findmax(K)[1])
        perms = permutations(dists, i)
        for j in perms
            push!(dist_poss, j)
        end
    end
    dist_poss = unique(sort.(dist_poss))
    dist_probs = fill(0.0, size(dist_poss,1), size(out["K"],1))
    for i in eachindex(K)
        if i >1
            K_curr = trunc(Int,K[i])
            dist_curr = sort(dist[1:K_curr,i])
            dist_ind = findfirst([dist_curr] .== dist_poss)
            dist_probs[:,i] = dist_probs[:,i-1]
            dist_probs[dist_ind,i] += 1
        end
    end
    dist_probs = dist_probs./sum(dist_probs, dims = 1)
    return dist_probs
end

function VS_roll(out, dist_prob, cov, dims)
    dist = transpose(out["dist"])
    x_skel = transpose(out["Sk_x"])
    K = length(dist_prob)
    inc_prob = zeros(K)
    n = 1
    i = 1
    for d in eachrow(dist)
        if filter(x -> x != 0.0, d) == dist_prob
            for k in 1:K
                x = copy(x_skel[i,(dims*(k-1) + 1):k*dims])
                if x[cov + 2] == 0.0
                    inc_prob[k] +=1
                end
            end
            n += 1
        end
        i += 1
    end
    inc_prob = inc_prob./n
    return inc_prob
end


function heat_smp_plot(out_p, out; norm = false)
    labels = []
    for i in eachindex(out_p["dist_poss"])
        lab = ""
        for j in eachindex(out_p["dist_poss"][i])
            if out_p["dist_poss"][i][j] == 1
                lab = lab*"W"
            elseif out_p["dist_poss"][i][j] == 2
                lab = lab*"L"
            end
        end
        push!(labels,lab)
    end
    ind = size(out_p["dist_poss"],1)
    if !norm
        heatmap(out["Trans"][ind:-1:1,1:ind], c = :greys, xticks = (1:ind, labels), yticks = (1:ind, labels[ind:-1:1]))
        vline!(0.5:(ind+0.5), c=:grey)
        hline!(0.5:(ind+0.5), c=:grey)
    else
        norm_mat = sum(eachcol(out["Trans"][ind:-1:1,1:ind])).*ones(ind,ind)
        heatmap(out["Trans"][ind:-1:1,1:ind]./norm_mat, c = :greys, xticks = (1:ind, labels), yticks = (1:ind, labels[ind:-1:1]))
        vline!(0.5:(ind+0.5), c=:grey)
        hline!(0.5:(ind+0.5), c=:grey)
    end
end

function plot_hazard(out, dims, t, covs; add = false)
    x_skel = transpose(out["Sk_x"])
    K_ = transpose(out["K"])
    dist = transpose(out["dist"])
    h = zeros(length(t),size(x_skel,1))
    for i in 1:size(x_skel,1)
        for k in 1:trunc(Int, K_[1,i])
            x = copy(x_skel[i,(dims*(k-1) + 1):k*dims])
            if dist[i,k] == 1
                h[:,i] .+= exp(dot(covs,x[2:end]) + x[1]) .*t.^(exp(x[1])-1)
            elseif dist[i,k] == 2
                #μ = exp(dot(covs,x[2:end]))
                #μ_ = t./μ
                #μ_α = μ_.^(exp(x[1]))
                #h[:,i] .+= ((exp(x[1])/μ).*(μ_).^(exp(x[1])-1))./(1 .+ μ_α)
                μ = exp(dot(covs,x[2:end]))
                h[:,i] .+= exp.(x[1] .+ (exp(x[1])-1).*log.(t) .- log.(μ^exp(x[1]) .+ t.^exp(x[1])))
            end
        end
    end
    hμ = mean.(eachrow(h))
    h05 = quantile.(eachrow(h),0.05)
    h95 = quantile.(eachrow(h),0.95)
    #skps = findall(!isnan(hμ))
    if !add
        plot(t, hμ, lc = :red)
        plot!(t, h05, lc = :red, linestyle = :dash)
        plot!(t, h95, lc = :red, linestyle = :dash)
    else
        plot!(t, hμ, lc = :blue)
        plot!(t, h05, lc = :blue, linestyle = :dash)
        plot!(t, h95, lc = :blue, linestyle = :dash)
    end
end

function plot_sub_hazard(out, dist_plot, t, covs, dims)
    x_skel = transpose(out["Sk_x"])
    dist = transpose(out["dist"])
    K = length(dist_plot)
    h = zeros(length(t),size(x_skel,1),K)
    i = 1
    j = 1
    for d in eachrow(dist)
        if sort(filter(x -> x != 0.0, d)) == dist_plot
            for k in 1:K
                x = copy(x_skel[i,(dims*(k-1) + 1):k*dims])
                if d[k] == 1
                    h[:,j,k] .= exp(dot(covs,x[2:end]) + x[1]) .*t.^(exp(x[1])-1)
                elseif d[k] == 2
                    μ = exp(dot(covs,x[2:end]))
                    h[:,j,k] .= exp.(x[1] .+ (exp(x[1])-1).*log.(t) .- log.(μ^exp(x[1]) .+ t.^exp(x[1])))
                end
            end
            j += 1
        end
        i += 1
    end
    h = h[:,1:(j-1),:]
    h1 = mean(eachcol(h[:,:,1]))
    p = plot(t, h1, lc = :red)
    if K == 1
        display(p)
        return
    end
    h1 = mean(eachcol(h[:,:,2]))
    p = plot!(t, h1, lc = :blue)
    if K == 2
        display(p)
        return
    end
    h1 = mean(eachcol(h[:,:,3]))
    p = plot!(t, h1, lc = :green)
    display(p)
end

function sort_out(out, dists, dat)
    # Takes 'out' the output of the PDMP sampler, returns model specific samples in sorted fashion
    x_skel = out["Sk_x"]
    dist = out["dist"]
    K_ = trunc.(Int,out["K"])
    dist_poss = []
    dists = repeat(dists,findmax(K_)[1])
    for i in 1:findmax(K_)[1]
        perms = permutations(dists, i)
        for j in perms
            push!(dist_poss, j)
        end
    end
    dist_poss = unique(sort.(dist_poss))
    dist_ind = []
    x_out = copy(x_skel)
    for i in axes(x_skel,2)
        #println(i)
        K_curr = K_[i]
        # Get distribution
        dist_curr = sort(dist[1:K_curr,i])
        push!(dist_ind, findfirst([dist_curr] .== dist_poss))
        # Sort by distribution
        ind_dist = sortperm(dist[1:K_curr,i])
        x_hold = []
        for k in 1:K_curr
            push!(x_hold, x_skel[((k-1)*(dat.p + 1) + 1):((k)*(dat.p + 1)),i])
        end
        x_hold = x_hold[ind_dist]
        # Sort Weibull
        ind_W = findall(dist_curr .== 1)
        if !isempty(ind_W)
            sort_W = sortperm(getindex.(x_hold[ind_W],1))
            x_hold[ind_W] = x_hold[ind_W][sort_W]
        end
        # Sort Log-logistic
        ind_LL = findall(dist_curr .== 2)
        if !isempty(ind_LL)
            sort_LL = sortperm(getindex.(x_hold[ind_LL],1))
            x_hold[ind_LL] = x_hold[ind_LL][sort_LL]    
        end
        x_out[1:((K_curr)*(dat.p + 1)),i] = reduce(vcat, x_hold)
    end
    output = vcat(transpose(K_), transpose(dist_ind), x_out)
    return Dict([("Out", output), ("Dists", dist_poss)])
end

function subhazard_quant(sorted_out, dist, covar, t)
    smps = sorted_out["Out"]
    dist_ind = findfirst([dist] .== sorted_out["Dists"])
    smps = smps[:,findall(smps[2,:] .== dist_ind)]
    K = length(dist)
    smps = smps[3:(2 + K*(length(covar) + 1)),:]
    for k in 1:K
        smps[((k-1)*(1 + length(covar)) + 2):(k)*(1 + length(covar)), :] = smps[((k-1)*(1 + length(covar)) + 2):(k)*(1 + length(covar)), :].*covar
    end
    plot_points = []
    h_ = Array{Float64}(undef, length(t), size(smps,2), K)
    S_ = Array{Float64}(undef, length(t), size(smps,2), K)
    for k in 1:K
        for i in axes(smps,2)
            μ_ = smps[((k-1)*(1 + length(covar)) + 1):(k)*(1 + length(covar)),i]
            if dist[k] == 1
                h_[:,i,k] = plot_Wei_h(μ_, t)
                S_[:,i,k] = plot_Wei_S(μ_, t)
            elseif dist[k] == 2
                h_[:,i,k] = plot_LL_h(μ_, t)
                S_[:,i,k] = plot_LL_S(μ_, t)
            else
                error("pls")
            end
        end
    end
    μ = mean(h_, dims = 2)
    μ_h = vec(mean(sum(h_, dims = 3), dims = 2))
    μ_S = vec(mean(exp.(sum(S_, dims = 3)), dims = 2))
    p_05 = Array{Float64}(undef, length(t), K)
    p_95 = Array{Float64}(undef, length(t), K)
    for k in 1:K
        p_05[:,k] = quantile.(eachrow(h_[:,:,k]), 0.05)
        p_95[:,k] = quantile.(eachrow(h_[:,:,k]), 0.95)
    end 
    plot_points = Array{Float64}(undef,length(t),3,K)
    for k in 1:K 
        plot_points[:,:,k] = hcat(μ[:,:,k],p_05[:,k],p_95[:,k])
    end
    return plot_points, μ_h, μ_S
end

function plot_Wei_h(μ_, t)
    h_μ = exp.(sum(μ_) .+ (exp(μ_[1])-1).*log.(t))
    return h_μ
end

function plot_Wei_S(μ_, t)
    S_μ = -exp(sum(μ_[2:end])).*t.^exp(μ_[1])
    return S_μ
end

function plot_LL_h(μ_, t)
    h_μ = exp.(μ_[1] .+ (exp(μ_[1])-1).*log.(t) .- log.(exp(sum(μ_[2:end]))^exp(μ_[1]) .+ t.^exp(μ_[1])))
    h_μ = min.(10_000, h_μ)
    return h_μ
end

function plot_LL_S(μ_, t)
    S_μ = log.((1 .+ (t./exp(sum(μ_[2:end]))).^exp(μ_[1])).^(-1))
    return S_μ
end

function hazard_quant(sorted_out, covar, t, smps_thresh)
    dist = sorted_out["Out"][2,:]
    dists = sorted_out["Dists"]
    dist_probs = zeros(length(dists))
    for j in eachindex(dist)
        dist_probs[dist[j]] += 1
    end
    dist_probs = dist_probs/sum(dist_probs)
    dist_probs[findall(dist_probs .< smps_thresh)] .= 0.0
    dist_probs = dist_probs/sum(dist_probs)
    h_μ = zeros(length(t), length(dists))
    lS_μ = zeros(length(t), length(dists))
    for d in eachindex(dists)
        if dist_probs[d] > 0.0
            dead, h_μ[:,d], lS_μ[:,d] = subhazard_quant(sorted_out, dists[d], covar, t)
        end
    end
    # Weight average of h_μ
    h_μ_ = sum(h_μ.*transpose(dist_probs), dims = 2)
    S_μ_ = sum(lS_μ.*transpose(dist_probs), dims = 2)
    return h_μ, h_μ_, S_μ_
end

function VS_out(sorted_out, dat)
    smps = sorted_out["Out"]
    dists = sorted_out["Dists"]
    VS_prob = zeros(dat.p - 1, length(dists[end]), length(dists))
    for d in eachindex(dists)
        d_smp = smps[3:end,findall(smps[2,:] .== d)]
        if size(d_smp,2) > 0.0
            K = length(dists[d])
            for k in 1:K
                k_smp = d_smp[(3 + (k-1)*(dat.p + 1)):((k)*(dat.p + 1)),:]
                VS_prob[:,k,d] = sum(k_smp .!= 0.0, dims = 2)/size(k_smp,2)
            end
        end
    end
    return VS_prob
end

function mean_survival(out, cov,  t_horizon, dat, iters, dists)
    # Computes mean survival given the samples from a polyhazard model "out"
    println("Computing mean survival given covariates");
    println(cov);
    x_skel = out["Sk_x"]
    dist = out["dist"]
    K_ = trunc.(Int,out["K"])
    t_store = zeros(size(x_skel,2), iters)
    K = out["K"]
    dist_poss = []
    dist = out["dist"]
    dists = repeat(dists,trunc(Int,findmax(K)[1]))
    for i in 1:trunc(Int,findmax(K)[1])
        perms = permutations(dists, i)
        for j in perms
            push!(dist_poss, j)
        end
    end
    dist_poss = unique(sort.(dist_poss))
    dist_t = zeros(size(x_skel,2), iters, length(dist_poss))
    for i in axes(x_skel, 2)
        K_curr = K_[i]
        dist_curr = dist[1:K_curr,i]
        dist_ind = findall([sort(dist_curr)] .== dist_poss)[1]
        for j in 1:iters
            t = zeros(K_curr)
            # Get distribution
            for k in 1:K_curr
                if dist_curr[k] == 1
                    # Weibull
                    γ = exp(x_skel[(k-1)*(dat.p + 1) + 1,i])
                    μ = exp(dot(x_skel[((k-1)*(dat.p + 1) + 2):(k*(dat.p + 1)),i],cov))
                    μ_ = μ^(-1/γ)
                    μ_ = exp(-(1/γ)*log(μ))
                    t_smp = rand(Weibull(γ, μ_))
                    if isnan(t_smp)
                        t_smp = Inf
                    end
                    t[k] = t_smp
                elseif dist_curr[k] == 2
                    γ = exp(x_skel[(k-1)*(dat.p + 1) + 1,i])
                    μ = exp(dot(x_skel[((k-1)*(dat.p + 1) + 2):(k*(dat.p + 1)),i],cov))
                    # No Distributions.jl so work with the cdf
                    u = rand()
                    t_smp = μ*(u\(1-u))^(1/γ)
                    if isnan(t_smp)
                        t_smp = Inf
                    end
                    t[k] = t_smp
                end
            end
            if minimum(t) < t_horizon
                t_store[i,j] = minimum(t)
                dist_t[i,j,dist_ind] =  minimum(t)
            else
                for k in 1:K_curr
                    if dist_curr[k] == 1
                        # Weibull
                        γ = exp(x_skel[(k-1)*(dat.p + 1) + 1,i])
                        μ = exp(dot(x_skel[((k-1)*(dat.p + 1) + 2):(k*(dat.p + 1)),i],cov))
                        μ_ = μ^(-1/γ)
                        t_smp = rand(Weibull(γ, μ_))
                        t[k] = t_smp
                        if isnan(t_smp)
                            println("Weibull");
                            println(γ);println(μ_)
                            
                        end
                    elseif dist_curr[k] == 2
                        γ = exp(x_skel[(k-1)*(dat.p + 1) + 1,i])
                        μ = exp(dot(x_skel[((k-1)*(dat.p + 1) + 2):(k*(dat.p + 1)),i],cov))
                        # No Distributions.jl so work with the cdf
                        u = rand()
                        t_smp = μ*(u\(1-u))^(1/γ)
                        t[k] = t_smp
                        if isnan(t_smp)
                            println("LL");
                            println(γ);println(μ)
                        end
                    end
                end
                if minimum(t) < t_horizon
                    t_store[i,j] = minimum(t)
                    dist_t[i,j,dist_ind] =  minimum(t)
                else
                    println(x_skel[:,i])
                    println(t)
                    println(i)
                    error("Bad horizon")
                end
            end
        end
    end
    dist_m = zeros(length(dist_poss))
    for i in eachindex(dist_poss)
        dist_m[i] = mean(dist_t[:,:,i][findall(dist_t[:,:,i] .!= 0.0)])
    end
    return Dict([("Overall",(mean(vec(t_store)), quantile(vec(t_store), 0.05), quantile(vec(t_store), 0.95))),
                    ("Model",dist_m),
                    ("Dist_poss", dist_poss)])
end

function mean_survival2(out, covar,  t, dists)
    x_skel = out["Sk_x"]
    dist = out["dist"]
    K_ = trunc.(Int,out["K"])
    K = out["K"]
    dist_poss = []
    dist = out["dist"]
    dists = repeat(dists,trunc(Int,findmax(K)[1]))
    for i in 1:trunc(Int,findmax(K)[1])
        perms = permutations(dists, i)
        for j in perms
            push!(dist_poss, j)
        end
    end
    dist_poss = unique(sort.(dist_poss))
    dist_store = zeros(size(x_skel,2))
    mean_surv = zeros(size(x_skel,2))
    for i in axes(x_skel, 2)
        K_curr = K_[i]
        dist_curr = dist[1:K_curr,i]
        dist_ind = findall([sort(dist_curr)] .== dist_poss)[1]
        # Get distribution
        l_S = zeros(K_curr, size(t,1))
        for k in 1:K_curr
            μ_ = x_skel[((k-1)*(1 + length(covar)) + 1):(k)*(1 + length(covar)),i].*vcat(1,covar)
            if dist_curr[k] == 1
                # Weibull
                l_S[k,:] = plot_Wei_S(μ_, t)
            elseif dist_curr[k] == 2
                # Log-logistic
                l_S[k,:] = plot_LL_S(μ_, t)
            end
        end
        dist_store[i] = copy(dist_ind)
        surv = exp.(sum(l_S, dims = 1))
        mean_surv[i] = step(t)*(sum(surv) - surv[begin] - surv[end])
    end
    return mean_surv, dist_store
end

function prior_predictive(σ1, σ2, dists, t, its)
    K = length(dists)
    α = rand(Normal(0.0,σ1),its, K)
    β = rand(Normal(0.0,σ2),its, K)
    surv = zeros(its)
    for i in 1:its
        l_S = zeros(K, size(t,1))
        for k in 1:K
            if dists[k] == 1
                l_S[k,:] = plot_Wei_S([α[i,k], β[i,k]],t)
            elseif dists[k] == 2
                l_S[k,:] = plot_LL_S([α[i,k], β[i,k]],t)
            end
        end
        S_ = exp.(sum(l_S, dims = 1))
        surv[i] = step(t)*(sum(S_) - S_[begin] - S_[end])
    end
    return surv
end