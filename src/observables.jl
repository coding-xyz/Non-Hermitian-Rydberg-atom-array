"""
    spinwave_states(model::model_nh)

Construct normalized open-boundary spin-wave states `|ψ_ℓ⟩` in the
single-excitation manifold (`ℓ = 1..Na`).
"""
function spinwave_states(model::model_nh)
    Na = model.Na
    ψl = Vector{Ket}(undef, Na)
    for l = 1:Na
        ψ = sum([sin(π * l * k / (Na + 1)) * model.Mge[k]' * model.ψ0 for k = 1:Na])
        ψl[l] = normalize(ψ)
    end
    return ψl
end

"""
    spinwave_projection_weights(model::model_nh, Ω, V, γ; normalize_rows=true, sort_by=:imag_desc)

Compute projection weights `|⟨L_α|ψ_ℓ⟩|^2` of spin-wave states onto
biorthogonal left eigenvectors of the full non-Hermitian Hamiltonian.
Returns a named tuple with `(weights, evals, sort_index)`.
"""
function spinwave_projection_weights(model::model_nh, Ω, V, γ; normalize_rows = true, sort_by = :imag_desc)
    evals, _, evecL = eigen_nh(model.H(Ω, V, γ))
    idx = if sort_by == :imag_desc
        sortperm(imag.(evals), rev = true)
    elseif sort_by == :imag_asc
        sortperm(imag.(evals), rev = false)
    else
        collect(eachindex(evals))
    end

    ψl = spinwave_states(model)
    W = zeros(Float64, length(ψl), length(evals))

    for l in eachindex(ψl), a in eachindex(evals)
        W[l, a] = abs2(dot(evecL[:, idx[a]], ψl[l].data))
    end

    if normalize_rows
        W = W ./ (maximum(W, dims = 2) .+ eps())
    end

    return (weights = W, evals = evals[idx], sort_index = idx)
end

"""
    projector_zero_one(model::model_nh)

Projector onto the `{0,1}`-excitation sector of a two-level full-Hilbert model.
"""
function projector_zero_one(model::model_nh)
    P01 = dm(model.ψ0)
    for i = 1:model.Na
        ψi = normalize(model.Mge[i]' * model.ψ0)
        P01 += dm(ψi)
    end
    return P01
end

"""
    sector_leakage(P01, ψ)

Compute `1 - ⟨P01⟩_norm`, i.e. weight outside the `{0,1}` sector.
"""
function sector_leakage(P01, ψ::Ket)
    den = real(ψ' * ψ)
    den == 0 && return 0.0
    return 1.0 - real((ψ' * (P01 * ψ)) / den)
end

"""
    sector_leakage_curve(t, model::model_nh, Ω, V, γ)

Time-resolved weight outside the `{0,1}` sector for full-Hilbert dynamics.
"""
function sector_leakage_curve(t, model::model_nh, Ω, V, γ)
    P01 = projector_zero_one(model)
    _, ψt = timeevolution.schroedinger(t, model.ψ0, model.H(Ω, V, γ))
    return [sector_leakage(P01, ψ) for ψ in ψt]
end

"""
    nearest_neighbor_connected(model::model_nh, ψ)

Average connected nearest-neighbor excitation correlation:
`mean_i(⟨n_i n_{i+1}⟩ - ⟨n_i⟩⟨n_{i+1}⟩)` with `n_i = |e⟩⟨e|_i`.
"""
function nearest_neighbor_connected(model::model_nh, ψ::Ket)
    if model.Na < 2
        return 0.0
    end
    den = real(ψ' * ψ)
    den == 0 && return 0.0

    C = 0.0
    for i = 1:model.Na-1
        nij = real((ψ' * ((model.Mee[i] * model.Mee[i + 1]) * ψ)) / den)
        ni = real((ψ' * (model.Mee[i] * ψ)) / den)
        nj = real((ψ' * (model.Mee[i + 1] * ψ)) / den)
        C += nij - ni * nj
    end
    return C / (model.Na - 1)
end

"""
    nearest_neighbor_connected_curve(t, model::model_nh, Ω, V, γ)

Time trace of connected nearest-neighbor correlations.
"""
function nearest_neighbor_connected_curve(t, model::model_nh, Ω, V, γ)
    _, ψt = timeevolution.schroedinger(t, model.ψ0, model.H(Ω, V, γ))
    return [nearest_neighbor_connected(model, ψ) for ψ in ψt]
end

"""
    compare_full_vs_sector01(Na, t, Ω, V, γ)

Compare full-Hilbert and strict zero+single-sector non-Hermitian dynamics.
Returns a named tuple with fidelity traces and mismatch diagnostics.
"""
function compare_full_vs_sector01(Na, t, Ω, V, γ)
    model_full = model_nh(Na)
    model_01 = model_nh_sector01(Na)

    fid_full, eval_full, pop_full = dynamics_nh(t, model_full, Ω, V, γ)
    fid_01, eval_01, pop_01 = dynamics_nh(t, model_01, Ω, V, γ)

    leak = sector_leakage_curve(t, model_full, Ω, V, γ)
    δfid = fid_full .- fid_01

    return (
        t = t,
        fidelity_full = fid_full,
        fidelity_sector01 = fid_01,
        fidelity_delta = δfid,
        leakage_ge2 = leak,
        evals_full = eval_full,
        evals_sector01 = eval_01,
        populations_full = pop_full,
        populations_sector01 = pop_01,
    )
end

# function correlation_ee(model::model_nh, ψ::Ket, r::Int; connected::Bool = true)
#     N = model.Na
#     (r < 1 || r >= N) && return 0.0
#     den = real(ψ' * ψ)
#     den == 0 && return 0.0

#     C = 0.0
#     for i = 1:N-r
#         j = i + r
#         corr = real((ψ' * ((model.Mee[i] * model.Mee[j]) * ψ)) / den)
#         if connected
#             ni  = real((ψ' * (model.Mee[i] * ψ)) / den)
#             nj  = real((ψ' * (model.Mee[j] * ψ)) / den)
#             corr -= ni * nj
#         end
#         C += corr
#     end
#     return C / (N - r)
# end

# # 平均距离 r 的 <σ_i^+ σ_{i+r}^->，可选 connected
# function correlation_pm(model::model_nh, ψ::Ket, r::Int; connected::Bool = true)
#     N = model.Na
#     (r < 1 || r >= N) && return 0.0 + 0.0im
#     den = real(ψ' * ψ)
#     den == 0 && return 0.0 + 0.0im

#     C = 0.0 + 0.0im
#     for i in 1:N-r
#         j = i + r
#         corr = (ψ' * ((model.Mge[i]' * model.Mge[j]) * ψ)) / den   # <σ_i^+ σ_j^->
#         if connected
#             si_p = (ψ' * (model.Mge[i]' * ψ)) / den                 # <σ_i^+>
#             sj_m = (ψ' * (model.Mge[j]  * ψ)) / den                 # <σ_j^->
#             corr -= si_p * sj_m
#         end
#         C += corr
#     end
#     return C / (N - r)
# end

# function correlations(model::model_nh, ψ::Ket, r::Int; connected::Bool = true)
#     N = model.Na
#     Mx = model.Mge .+ dagger.(model.Mge)
#     My = -im * (model.Mge .- dagger.(model.Mge) )
#     Mz = model.Mgg .- dagger.(model.Mee)
#     den = real(ψ' * ψ)
#     den == 0 && return 0.0 + 0.0im

#     Cx = 0.0 + 0.0im
#     Cy = 0.0 + 0.0im
#     Cz = 0.0 + 0.0im

#     s_x = [(ψ' * (Mx[i] * ψ)) / den for i in 1:N]
#     s_y = [(ψ' * (My[i] * ψ)) / den for i in 1:N]
#     s_z = [(ψ' * (Mz[i] * ψ)) / den for i in 1:N]

#     if r == 0
#         if connected
#             Cx = sum(1 .- s_x.^2) 
#             Cy = sum(1 .- s_y.^2)
#             Cz = sum(1 .- s_z.^2)
#         else
#             Cx = N + 0.0im
#             Cy = N + 0.0im
#             Cz = N + 0.0im
#         end
#     else
#         for i in 1:N-r
#             j = i + r
#             corr_x = (ψ' * ((Mx[i] * Mx[j]) * ψ)) / den
#             corr_y = (ψ' * ((My[i] * My[j]) * ψ)) / den
#             corr_z = (ψ' * ((Mz[i] * Mz[j]) * ψ)) / den
#             if connected
#                 corr_x -= s_x[i] * s_x[j]
#                 corr_y -= s_y[i] * s_y[j]
#                 corr_z -= s_z[i] * s_z[j]
#             end
#             Cx += corr_x
#             Cy += corr_y
#             Cz += corr_z
#         end
#     end
#     return Cx / (N - r), Cy / (N-r), Cz / (N-r)
# end

# bipartite entanglement entropy S_A(t), 默认 cut = N/2
ent_entropy_state(model::model_nh, ψ::Ket; cut::Int = fld(model.Na, 2), base::Real = exp(1)) =
    entanglement_entropy(ψ.data, model.Na; cut = cut, local_dim = 2, base = base)


function correlations(model::model_nh, ψ_t::AbstractVector{<:Ket}; connected::Bool = true)
    N = model.Na
    nt = length(ψ_t)

    Mx = model.Mge .+ dagger.(model.Mge)
    My = -im .* (model.Mge .- dagger.(model.Mge))
    Mz = model.Mgg .- model.Mee

    Cx = zeros(Float64, nt)
    Cy = zeros(Float64, nt)
    Cz = zeros(Float64, nt)

    for k in eachindex(ψ_t)
        ψ = ψ_t[k]
        den = real(ψ' * ψ)
        den == 0 && continue

        sx = [(ψ' * (Mx[i] * ψ)) / den for i in 1:N]
        sy = [(ψ' * (My[i] * ψ)) / den for i in 1:N]
        sz = [(ψ' * (Mz[i] * ψ)) / den for i in 1:N]

        for m in 1:N, n in 1:N
            cx = (ψ' * ((Mx[m] * Mx[n]) * ψ)) / den
            cy = (ψ' * ((My[m] * My[n]) * ψ)) / den
            cz = (ψ' * ((Mz[m] * Mz[n]) * ψ)) / den

            if connected
                cx -= sx[m] * sx[n]
                cy -= sy[m] * sy[n]
                cz -= sz[m] * sz[n]
            end

            Cx[k] += real(cx)
            Cy[k] += real(cy)
            Cz[k] += real(cz)
        end

        Cx[k] /= N^2
        Cy[k] /= N^2
        Cz[k] /= N^2
    end

    return Cx, Cy, Cz
end