"""
    counting_single(filename)

Aggregate single-atom counting data from CSV and return `(t, y, y_std, p0)`.
"""
function counting_single(filename)
    df = CSV.read(filename, DataFrame)

    t = []
    data = []

    for row in eachrow(df)
        ti = row["time"]
        if ismissing(ti)
            continue
        elseif ti in t
            idx = findfirst(ti .== t)
            data[idx] = data[idx] .+ [row["N0"], row["N1"]]
        else
            push!(t, ti)
            push!(data, [row["N0"], row["N1"]])
        end
    end

    p_mean = [n[2] / sum(n) for n in data]
    y = p_mean ./ p_mean[1]
    y_std = [sqrt(y[i] * (1 - y[i]) / sum(data[i])) for i in eachindex(data)]

    return t, y, y_std, p_mean[1]
end

"""
    counting_double(filename)

Aggregate two-atom counting data from CSV and return `(t, y, y_std, p0)`.
"""
function counting_double(filename)
    df = CSV.read(filename, DataFrame)

    t = []
    data = []

    for row in eachrow(df)
        ti = row["time"]
        if ismissing(ti)
            continue
        elseif ti in t
            idx = findfirst(ti .== t)
            data[idx] = data[idx] .+ [row["N00"], row["N01"], row["N10"], row["N11"]]
        else
            push!(t, ti)
            push!(data, [row["N00"], row["N01"], row["N10"], row["N11"]])
        end
    end

    p_mean = [n[4] / sum(n) for n in data]
    y = p_mean ./ p_mean[1]
    y_std = [sqrt(y[i] * (1 - y[i]) / sum(data[i])) for i in eachindex(data)]

    return t, y, y_std, p_mean[1]
end

"""
    bayes_binomial(filename, x)

Compute Bayesian binomial posteriors over grid `x` from counting CSV data.
"""
function bayes_binomial(filename, x)
    df = CSV.read(filename, DataFrame)

    t = []
    data = Vector{Vector{Int64}}[]

    for row in eachrow(df)
        ti = row["time"]
        if ismissing(ti)
            continue
        elseif ti in t
            idx = findfirst(ti .== t)
            push!(data[idx], [row["N0"]; row["N1"]])
        else
            push!(t, ti)
            push!(data, [[row["N0"]; row["N1"]]])
        end
    end

    P = ones(length(x), length(t)) ./ length(x)
    P_mean = zeros(length(t))

    for i = 1:length(t)
        for (n0, n1) in data[i]
            P[:, i] = @. x^n1 * (1 - x)^n0 * P[:, i]
            P[:, i] = P[:, i] ./ sum(0.01 * log(10) * x .* P[:, i])
        end
        P_mean[i] = sum(0.01 * log(10) * x .^ 2 .* P[:, i])
    end

    return t, P, P_mean
end
