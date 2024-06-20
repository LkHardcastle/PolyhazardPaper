function ∇U(x::Matrix{Float64}, K::Int64, dist::Vector{Int64}, priors::Prior, dat::PolyData)
    # Returns an matrix of partial derivatives ∇Ux
    # Store gradients
    ∇Ux = zeros(dat.p + 1, K)
    # Gradient of the potential
    haz = fill(Inf64, K)
    # Need to check this
    ∇h = fill(Inf64, dat.p + 1, K)
    dh_h = fill(Inf64, dat.p + 1, K)
    #U_store = zeros(dat.n, dat.p + 1, K)
    for i in 1:dat.n
        for k in 1:K
            if dist[k] == 1
                # Weibull
                haz[k], ∇h[:,k], ∇H, dh_h[:,k] = W_eval(x[:,k], dat, i)
            end
            if dist[k] == 2
                # Log-logistic
                haz[k], ∇h[:,k], ∇H, dh_h[:,k] = LL_eval(x[:,k], dat, i)
            end
            if dist[k] == 3
                # Gompertz
                haz[k], ∇h[:,k], ∇H, dh_h[:,k] = Go_eval(x[:,k], dat, i)
            end
            if dist[k] == 4
                # Gamma
            end
            if dist[k] == 5
                # Log-normal
            end
            # Shape grad
            ∇Ux[1,k] += ∇H[1]
            #U_store[i,1,k] += ∇H[1]
            # Coefficient grad 
            for j in 1:dat.p
                ∇Ux[j+1,k] += ∇H[j+1]
                #U_store[i,j+1,k] += ∇H[j+1]
            end
        end
        if dat.cens[i] == 1
            haz_norm = sum(haz)
            for k in 1:K
                # Shape grad
                ∇Ux[1,k] -= ∇h[1,k]/haz_norm
                #U_store[i,1,k] -= ∇h[1,k]/haz_norm
                # Coefficient grad 
                for j in 1:dat.p
                    ∇Ux[j+1,k] -= ∇h[j+1,k]/haz_norm
                    #U_store[i,j+1,k] = ∇h[j+1,k]/haz_norm
                end
            end
        end
    end
    # Priors
    prior_grad!(x, priors, ∇Ux, K)
    return ∇Ux
end


function U_birth(x::Matrix{Float64}, K::Int64, dist::Vector{Int64}, priors::Prior, dat::PolyData, dyn::Dynamics)
    U1 = 0.0
    U2 = 0.0
    # Gradient of the potential
    haz = zeros(K)
    # Need to check this
    H = zeros(K)
    for i in 1:dat.n
        for k in 1:K
            if dist[k] == 1
                # Weibull
                haz[k], H[k] = W_eval(x[:,k], dat, i; grad = false)
            end
            if dist[k] == 2
                # Log-logistic
                haz[k], H[k] = LL_eval(x[:,k], dat, i; grad = false)
            end
            if dist[k] == 3
                # Gompertz
                haz[k], H[k] = Go_eval(x[:,k], dat, i; grad = false)
            end
            if dist[k] == 4
                # Gamma
            end
            if dist[k] == 5
                # Log-normal
            end
        end
        ## Innovation term
        if dyn.u_dist == 1
            # Weibull
            haz_u, H_u = W_eval(dyn.u, dat, i; grad = false)
        end
        if dyn.u_dist == 2
            # Log-logistic
            haz_u, H_u = LL_eval(dyn.u, dat, i; grad = false)
        end
        if dyn.u_dist == 3
            # Gompertz
            haz_u, H_u = Go_eval(dyn.u, dat, i; grad = false)
        end
        if dyn.u_dist == 4
            # Gamma
        end
        if dyn.u_dist == 5
            # Log-normal
        end

        U1 += dat.cens[i]*log(sum(haz)) - sum(H)
        U2 += dat.cens[i]*log(sum(haz) + haz_u) - sum(H) - H_u
    end
    for k in 1:K
        p_eval = prior_eval(x[:,k], priors)
        U1 += p_eval
        U2 += p_eval
    end
    U2 += log(priors.K_prob[K+1]) - log(K+1) 
    U1 += log(priors.K_prob[K]) - log(length(priors.dists))
    ## Note - while innovations are drawn from the prior distribution the innovation density evaluation cancels with the prior density evaluation in the MHG ratio
    if isinf(U1) && isinf(U2)
        return -10_000
    end
    if isnan(U2)
        return -10_000
    elseif isnan(U1)
        return 10_000
    else
        return U2 - U1
    end
end

function U_death(x::Matrix{Float64}, K::Int64, dist::Vector{Int64}, priors::Prior, dat::PolyData, dyn::Dynamics)
    U1 = 0.0
    U2 = 0.0
    # Gradient of the potential
    haz = zeros(K)
    # Need to check this
    H = zeros(K)
    for i in 1:dat.n
        for k in 1:K
            if dist[k] == 1
                # Weibull
                haz[k], H[k] = W_eval(x[:,k], dat, i; grad = false)
            end
            if dist[k] == 2
                # Log-logistic
                haz[k], H[k] = LL_eval(x[:,k], dat, i; grad = false)
            end
            if dist[k] == 3
                # Gompertz
                haz[k], H[k] = Go_eval(x[:,k], dat, i; grad = false)
            end
            if dist[k] == 4
                # Gamma
            end
            if dist[k] == 5
                # Log-normal
            end
        end
        U1 += dat.cens[i]*log(sum(haz)) - sum(H)
        U2 += dat.cens[i]*log(sum(haz[1:end .!= dyn.h_death])) - sum(H[1:end .!= dyn.h_death])

    end
    for k in 1:K
        # Probably a better way of writing but doesn't matter too much
        if k != dyn.h_death
            p_eval = prior_eval(x[:,k], priors)
            U1 += p_eval
            U2 += p_eval
        end
    end
    U2 += log(priors.K_prob[K-1]) - log(length(priors.dists))
    U1 += log(priors.K_prob[K]) - log(K)
    ## Note - while innovations are drawn from the prior distribution the innovation density evaluation cancels with the prior density evaluation in the MHG ratio
    if isinf(U1) && isinf(U2)
        return -10_000
    end
    if isnan(U2)
        return -10_000
    elseif isnan(U1)
        return 10_000
    else
        return U2 - U1
    end
    return U2 - U1
end

function U_swap(x::Matrix{Float64}, K::Int64, dist::Vector{Int64}, priors::Prior, dat::PolyData, dyn::Dynamics)
    U1 = 0.0
    U2 = 0.0
    # Gradient of the potential
    haz = zeros(K)
    # Need to check this
    H = zeros(K)
    for i in 1:dat.n
        for k in 1:K
            if dist[k] == 1
                # Weibull
                haz[k], H[k] = W_eval(x[:,k], dat, i; grad = false)
            end
            if dist[k] == 2
                # Log-logistic
                haz[k], H[k] = LL_eval(x[:,k], dat, i; grad = false)
            end
            if dist[k] == 3
                # Gompertz
                haz[k], H[k] = Go_eval(x[:,k], dat, i; grad = false)
            end
            if dist[k] == 4
                # Gamma
            end
            if dist[k] == 5
                # Log-normal
            end
        end
        U1 += dat.cens[i]*log(sum(haz)) - sum(H)
        if dyn.swap_dist == 1
            # Weibull
            haz[dyn.h_swap], H[dyn.h_swap] = W_eval(dyn.u, dat, i; grad = false)
        end
        if dyn.swap_dist == 2
            # Log-logistic
            haz[dyn.h_swap], H[dyn.h_swap] = LL_eval(dyn.u, dat, i; grad = false)
        end
        if dyn.swap_dist == 3
            # Gompertz
            haz[dyn.h_swap], H[dyn.h_swap] = Go_eval(dyn.u, dat, i; grad = false)
        end
        if dyn.swap_dist == 4
            # Gamma
        end
        if dyn.swap_dist == 5
            # Log-normal
        end
        U2 += dat.cens[i]*log(sum(haz)) - sum(H)
    end
    p_eval = prior_eval(x[:,dyn.h_swap], priors)
    U1 += p_eval
    p_eval = prior_eval(dyn.u, priors)
    U2 += p_eval
    if isinf(U1) && isinf(U2)
        return -10_000
    end
    if isnan(U2)
        return -10_000
    elseif isnan(U1)
        return 10_000
    else
        return U2 - U1
    end
    return U2 - U1
end

function U_swap_all(x::Matrix{Float64}, K::Int64, dist::Vector{Int64}, priors::Prior, dat::PolyData, dyn::Dynamics)
    U1 = 0.0
    U2 = 0.0
    # Gradient of the potential
    haz = zeros(K)
    # Need to check this
    H = zeros(K)
    for i in 1:dat.n
        for k in 1:K
            if dist[k] == 1
                # Weibull
                haz[k], H[k] = W_eval(x[:,k], dat, i; grad = false)
            end
            if dist[k] == 2
                # Log-logistic
                haz[k], H[k] = LL_eval(x[:,k], dat, i; grad = false)
            end
            if dist[k] == 3
                # Gompertz
                haz[k], H[k] = Go_eval(x[:,k], dat, i; grad = false)
            end
            if dist[k] == 4
                # Gamma
            end
            if dist[k] == 5
                # Log-normal
            end
        end
        U1 += dat.cens[i]*log(sum(haz)) - sum(H)
    end
    # Gradient of the potential
    haz = zeros(K)
    # Need to check this
    H = zeros(K)
    for i in 1:dat.n
        for k in 1:K
            if dyn.dist_all[k] == 1
                # Weibull
                haz[k], H[k] = W_eval(dyn.u_all[:,k], dat, i; grad = false)
            end
            if dyn.dist_all[k] == 2
                # Log-logistic
                haz[k], H[k] = LL_eval(dyn.u_all[:,k], dat, i; grad = false)
            end
            if dyn.dist_all[k] == 3
                # Gompertz
                haz[k], H[k] = Go_eval(dyn.u_all[:,k], dat, i; grad = false)
            end
            if dyn.dist_all[k] == 4
                # Gamma
            end
            if dyn.dist_all[k] == 5
                # Log-normal
            end
        end
        U2 += dat.cens[i]*log(sum(haz)) - sum(H)
    end
    if isinf(U1) && isinf(U2)
        return -10_000
    end
    if isnan(U2)
        return -10_000
    elseif isnan(U1)
        return 10_000
    else
        return U2 - U1
    end
    return U2 - U1
end

function prior_eval(x::Array{Float64}, priors::Prior)
    ## Evaluation of U_p
    return logpdf(Normal(0,priors.σ_α), x[1]) + logpdf(Normal(0,priors.σ_β0), x[2]) + sum(log.(pdf.(Normal(0,priors.σ_β), x[3:end]).*(1-priors.w_vs).*(x[3:end] .!= 0.0) .+ priors.w_vs.*(x[3:end] .== 0.0)))
end

function prior_eval(dyn::Dynamics, priors::Prior)
    ## Evaluation of U_p
    return logpdf(Normal(0,priors.σ_α), dyn.u[1]) + logpdf(Normal(0,priors.σ_β0), dyn.u[2]) + sum(log.(pdf.(Normal(0,priors.σ_β), dyn.u[3:end]).*(1-priors.w_vs).*(dyn.u[3:end] .!= 0.0) .+ priors.w_vs.*(dyn.u[3:end] .== 0.0)))
end

function prior_grad!(x::Array{Float64}, priors::Prior, ∇Ux::Matrix{Float64}, K::Int)
    ## Evaluation of ∇U_p
    for k in 1:K
        ∇Ux[1,k] += x[1,k]/priors.σ_α^2
        ∇Ux[2, k] += x[2,k]/priors.σ_β0^2
        ∇Ux[3:end, k] .+= x[3:end, k]/priors.σ_β^2
    end
end

function prior_eval(x::Array{Float64}, priors::PriorNoVS)
    ## Evaluation of U_p
    return logpdf(Normal(0,priors.σ_α), x[1]) + logpdf(Normal(0,priors.σ_β0), x[2]) + sum(logpdf.(Normal(0,priors.σ_β), x[3:end]))
end

function prior_eval(dyn::Dynamics, priors::PriorNoVS)
    ## Evaluation of U_p
    return logpdf(Normal(0,priors.σ_α), dyn.u[1]) + logpdf(Normal(0,priors.σ_β0), dyn.u[2]) + sum(logpdf.(Normal(0,priors.σ_β), dyn.u[3:end]))
end



function W_eval(x::Vector{Float64}, dat::PolyData, i::Int64 ;grad = true)
    # Returns the corresponding hazard
    # A vector of partial derivatives of the hazard
    # A vector of partial derivatives of the cumulative hazard
    y = dat.y[i]
    covs = dat.cov[:,i]
    if grad
        ∇h = zeros(dat.p + 1)
        ∇H = zeros(dat.p + 1)
        dh_h = zeros(dat.p + 1)
        h = exp(dot(covs,x[2:end]) + x[1])*y^(exp(x[1])-1)
        dh_h[1] = (exp(x[1])*log(y) + 1)
        ∇h[1] = h*dh_h[1]
        ∇H[1] = h*y*log(y)
        for j in 2:(dat.p + 1)
            if covs[j-1] == 0.0
                ∇h[j] = 0.0
                ∇H[j] = 0.0
            else
                dh_h[j] = covs[j-1]
                ∇h[j] = covs[j-1]*h
                ∇H[j] = covs[j-1]*h*exp(-x[1])*y
            end
            
        end
        return h, ∇h, ∇H, dh_h
    end
    if !grad
        h = exp(dot(covs,x[2:end]) + x[1])*y^(exp(x[1])-1)
        H = h*exp(-x[1])*y
        return h, H
    end
end

function LL_eval(x::Vector{Float64}, dat::PolyData, i::Int64 ;grad = true)
    # Returns the corresponding hazard
    # A vector of partial derivatives of the hazard
    # A vector of partial derivatives of the cumulative hazard
    y = dat.y[i]
    covs = dat.cov[:,i]
    μ = exp(dot(covs,x[2:end]))
    μ_ = y/μ
    μ_α = μ_^(exp(x[1]))
    if grad
        ∇h = zeros(dat.p + 1)
        ∇H = zeros(dat.p + 1)
        dh_h = zeros(dat.p + 1)
        #h = exp(x[1] - log(μ) + (exp(x[1])-1)*log(μ_) - log(1 + μ_α))
        h = exp(x[1] + (exp(x[1])-1)*log(y) - log(μ^exp(x[1]) + y^exp(x[1])))
        dh_h[1] = ((1 + exp(x[1])*log(μ_)) - log(μ_)*exp(-log(1 + μ_α^-1) + x[1]))
        ∇h[1] = h*dh_h[1]
        ∇H[1] = h*μ_*log(μ_)*μ
        h_add = exp(2*x[1] + (exp(x[1])-1)*log(y) + exp(x[1])*log(μ) - 2*log(μ^exp(x[1]) + y^exp(x[1])))
        dh_h_add = exp(x[1] + exp(x[1])*log(μ) - log(μ^exp(x[1]) + y^exp(x[1])))
        for j in 2:(dat.p + 1)
            if covs[j-1] == 0.0
                ∇h[j] = 0.0
                ∇H[j] = 0.0
            else
                #∇h[j] = -covs[j-1]*exp(2*x[1] - log(y) - 2*log(μ_α^0.5 + μ_α^-0.5))
                ∇h[j] = -covs[j-1]*h_add
                dh_h[j] = -covs[j-1]*dh_h_add
                ∇H[j] = -covs[j-1]*h*y
            end
        end
        return h, ∇h, ∇H, dh_h
    end
    if !grad
        h = exp(x[1] + (exp(x[1])-1)*log(y) - log(μ^exp(x[1]) + y^exp(x[1])))
        H = log(1 + μ_α)
        return h, H
    end
end

function Go_eval(x::Vector{Float64}, dat::PolyData, i::Int64 ;grad = true)
    # Returns the corresponding hazard
    # A vector of partial derivatives of the hazard
    # A vector of partial derivatives of the cumulative hazard
    y = dat.y[i]
    covs = dat.cov[:,i]
    μ = exp(dot(covs,x[2:end]))
    if grad
        ∇h = zeros(dat.p + 1)
        ∇H = zeros(dat.p + 1)
        dh_h = zeros(dat.p + 1)
        h = exp(log(μ) + exp(x[1] + log(y)))
        H = exp(log(μ) - x[1] + (log(exp(exp(x[1])*y) - 1)))
        dh_h[1] = y*exp(x[1])
        ∇h[1] = h*dh_h[1]
        ∇H[1] = (1/exp(x[1]))*(h*(y*exp(x[1]) - 1) + μ)
        for j in 2:(dat.p + 1)
            if covs[j-1] == 0.0
                ∇h[j] = 0.0
                ∇H[j] = 0.0
            else
                dh_h[j] = covs[j-1]
                ∇h[j] = covs[j-1]*h
                ∇H[j] = covs[j-1]*H
            end
        end
        return h, ∇h, ∇H, dh_h
    end
    if !grad
        h = μ*exp(exp(x[1])*y)
        H = (μ/exp(x[1]))*(exp(exp(x[1])*y) - 1)
        return h, H
    end
    
end