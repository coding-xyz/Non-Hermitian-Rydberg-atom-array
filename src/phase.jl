"""
    phase(model::model_nh, Ω, V, γ)

Compute the phase metric `β` for a `model_nh` instance.
"""
function phase(model::model_nh, Ω, V, γ)
    return phase(model.H(Ω, V, γ))
end

"""
    phase(H_nH)

Compute the phase metric `β` from a non-Hermitian Hamiltonian.
"""
function phase(H_nH)
    λ, ϕR, ϕL = eigen_nh(dense(H_nH).data)
    λi = imag.(λ)
    β = 1 - maximum(λi) / mean(λi)
    return β
end

"""
    phase(H_nH, ψref)

Compute `(β, fm, ϵ)` phase diagnostics using a reference state `ψref`.
"""
function phase(H_nH, ψref)
    λ, ϕR, ϕL = eigen_nh(dense(H_nH).data)
    λi = imag.(λ)
    fid = abs.(ϕL' * ψref.data) .^ 2
    β = 1 - maximum(λi) / mean(λi)
    idx = findall((λi .- mean(λi)) .> 1e-3)
    if length(idx) == 0
        fm, ϵ = 0, 0
    elseif length(idx) == 1
        fm = fid[idx[1]]
        ϵ = 0
    else
        fid = sort(fid[idx], rev = true)
        fm = fid[1]
        ϵ = fid[2] / fid[1]
    end
    return β, fm, ϵ
end

function _sort_eigenpairs(evals, evecs; sort_by = :imag_desc)
    idx = if sort_by == :imag_desc
        sortperm(imag.(evals), rev = true)
    elseif sort_by == :imag_asc
        sortperm(imag.(evals), rev = false)
    elseif sort_by == :real_desc
        sortperm(real.(evals), rev = true)
    elseif sort_by == :real_asc
        sortperm(real.(evals), rev = false)
    else
        collect(eachindex(evals))
    end
    return evals[idx], evecs[:, idx], idx
end

"""
    entanglement_entropy(ψ::AbstractVector, Na; cut=fld(Na, 2), local_dim=2, base=2)

Compute the bipartite entanglement entropy of a pure state vector `ψ` for a
chain of `Na` sites with local dimension `local_dim`.

The chain is split as `1:cut | cut+1:Na`. For odd `Na`, the default
`cut = fld(Na, 2)` gives the balanced partition `floor(Na/2) | ceil(Na/2)`.
"""
function entanglement_entropy(ψ::AbstractVector, Na::Integer;
                              cut::Integer = fld(Na, 2),
                              local_dim::Integer = 2,
                              base::Real = exp(1))
    0 < cut < Na || throw(ArgumentError("cut must satisfy 0 < cut < Na"))
    length(ψ) == local_dim^Na || throw(ArgumentError("state length does not match local_dim^Na"))

    ψn = ψ / norm(ψ)
    ψmat = reshape(ψn, local_dim^cut, local_dim^(Na - cut))
    s = svdvals(ψmat)
    p = real.(s .^ 2)
    p = p[p .> eps(real(eltype(p)))]
    logbase = log(base)
    return -sum(p .* (log.(p) ./ logbase))
end

"""
    eigenstate_entanglement_entropy(model::model_nh, Ω, V, γ; cut=fld(model.Na, 2), sort_by=:imag_desc)

Compute bipartite entanglement entropy for all right eigenstates of the
non-Hermitian Hamiltonian. Returns a named tuple with
`(entropy, evals, sort_index)`.

For odd `model.Na`, the default partition is `floor(Na/2) | ceil(Na/2)`.
"""
function eigenstate_entanglement_entropy(model::model_nh, Ω, V, γ;
                                         cut::Integer = fld(model.Na, 2),
                                         sort_by::Symbol = :imag_desc)
    return eigenstate_entanglement_entropy(
        model.H(Ω, V, γ), model.Na;
        cut = cut,
        sort_by = sort_by,
    )
end

"""
    eigenstate_entanglement_entropy(H_nH, Na; cut=fld(Na, 2), sort_by=:imag_desc, local_dim=2)

Compute bipartite entanglement entropy for all right eigenstates of a
non-Hermitian Hamiltonian matrix or operator on a chain of `Na` sites.

Returns a named tuple with `(entropy, evals, sort_index)`.
"""
function eigenstate_entanglement_entropy(H_nH, Na::Integer;
                                         cut::Integer = fld(Na, 2),
                                         sort_by::Symbol = :imag_desc,
                                         local_dim::Integer = 2)
    Hmat = typeof(H_nH) <: Operator ? dense(H_nH).data : H_nH
    evals, evecs = eigen(Hmat)
    evals_sorted, evecs_sorted, idx = _sort_eigenpairs(evals, evecs; sort_by = sort_by)
    entropy = [
        entanglement_entropy(view(evecs_sorted, :, i), Na; cut = cut, local_dim = local_dim)
        for i in axes(evecs_sorted, 2)
    ]
    return (entropy = entropy, evals = evals_sorted, sort_index = idx)
end
