function VS_update!(v::Array{Float64}, VS_times::Array{Float64}, priors::PriorHyper)
    j = findmin(VS_times)[2]
    if v[j] == 0.0
        # Unstick
        v[j] = rand(Bernoulli(0.5))*2 - 1.0
        VS_times[j] = Inf64
    else
        # Stick
        v[j] = 0.0
        VS_times[j] = rand(Exponential(priors.w_vs/(1-priors.w_vs)))
    end
end
function VS_time_start(x::Array{Float64}, v::Array{Float64}, dat::PolyData, K::Int64)
    VS_times = zeros(dat.p + 1, K)
    for i in eachindex(x)
        if v[i] == 0.0
            VS_times[i] = rand(Exponential(priors.w_vs/(1-priors.w_vs)))
        elseif x[i]*v[i] < 0.0
            VS_times[i] = abs(x[i])
        else
            VS_times[i] = Inf
        end
    end
    VS_times[1:2,:] .= Inf
    return VS_times
end
