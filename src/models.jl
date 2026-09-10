"""
    model_nh

Container for the non-Hermitian two-level chain model.
"""
mutable struct model_nh
    Na::Int64
    basis
    ψ0::Ket
    Mgg::Vector{Operator}
    Mee::Vector{Operator}
    Mge::Vector{Operator}
    Vex::Vector{Operator}
    H::Function
end

"""
    model_nh(Na)

Build a long-range (`1/r^3`) non-Hermitian model for `Na` atoms.
Set `sigma` to enable position noise.
"""
function model_nh(Na; sigma = 0.0)
    b1 = NLevelBasis(2)
    bN = b1 ^ Na
    ψ0 = basisstate(bN, 1)
    if Na == 1
        Mge = [transition(b1, 1, 2)]
        Mgg = [transition(b1, 1, 1)]
        Mee = [transition(b1, 2, 2)]
        Vex = []
        H = (Ω, V, γ) -> Ω / 2 * sum(Mge + dagger.(Mge)) - im * γ / 2 * sum(Mee)
    else
        Mge = [embed(bN, i, transition(b1, 1, 2)) for i = 1:Na]
        Mgg = [embed(bN, i, transition(b1, 1, 1)) for i = 1:Na]
        Mee = [embed(bN, i, transition(b1, 2, 2)) for i = 1:Na]
        Vex_x = x -> [sum([abs(x[i] - x[j])^(-3) * (Mge[i] * Mge[j]' + Mge[j] * Mge[i]') for j = i+1:Na]) for i = 1:Na-1]
        Vex = Vex_x(1:Na)
        H = (Ω, V, γ) -> Ω / 2 * sum(Mge + dagger.(Mge)) +
                         V * sum(Vex_x(collect(1:Na) .+ sigma * randn(Na))) -
                         im * γ / 2 * sum(Mee)
    end
    return model_nh(Na, bN, ψ0, Mgg, Mee, Mge, Vex, H)
end

"""
    model_nh_nn(Na)

Build a nearest-neighbor non-Hermitian model for `Na` atoms.
Set `sigma` to enable random-coupling noise.
"""
function model_nh_nn(Na; sigma = 0.0)
    b1 = NLevelBasis(2)
    bN = b1 ^ Na
    ψ0 = basisstate(bN, 1)
    if Na == 1
        Mge = [transition(b1, 1, 2)]
        Mgg = [transition(b1, 1, 1)]
        Mee = [transition(b1, 2, 2)]
        Vex = []
        H = (Ω, V, γ) -> Ω / 2 * sum(Mge + dagger.(Mge)) - im * γ / 2 * sum(Mee)
    else
        Mge = [embed(bN, i, transition(b1, 1, 2)) for i = 1:Na]
        Mgg = [embed(bN, i, transition(b1, 1, 1)) for i = 1:Na]
        Mee = [embed(bN, i, transition(b1, 2, 2)) for i = 1:Na]
        Vex = [Mge[i] * Mge[i + 1]' + Mge[i + 1] * Mge[i]' for i = 1:Na-1]
        H = (Ω, V, γ) -> Ω / 2 * sum(Mge + dagger.(Mge)) +
                         V * sum((1 .+ sigma * randn(Na - 1)) .* Vex) -
                         im * γ / 2 * sum(Mee)
    end
    return model_nh(Na, bN, ψ0, Mgg, Mee, Mge, Vex, H)
end

"""
    model_me

Container for the three-level master-equation model.
"""
mutable struct model_me
    Na::Int64
    basis
    ψ0::Ket
    Mgg::Vector{Operator}
    Mee::Vector{Operator}
    Mdd::Vector{Operator}
    Mge::Vector{Operator}
    Mde::Vector{Operator}
    Vex::Vector{Operator}
    H::Function
    J::Function
end

"""
    model_me(Na; interaction=:nn)

Build a three-level (`g`, `e`, `d`) model and collapse operators for `Na` atoms.
`interaction` can be `:nn` or `:C3` (`1/r^3`).
"""
function model_me(Na; interaction::Symbol = :nn)
    interaction ∉ (:nn, :C3) && throw(ArgumentError("interaction must be :nn or :C3"))

    b1 = NLevelBasis(3)
    bN = b1 ^ Na
    ψ0 = basisstate(bN, 1)

    if Na == 1
        Mgg = [transition(b1, 1, 1)]
        Mee = [transition(b1, 2, 2)]
        Mdd = [transition(b1, 3, 3)]
        Mge = [transition(b1, 1, 2)]
        Mde = [transition(b1, 3, 2)]
        Vex = []
        H = (Ω, V) -> Ω / 2 * sum(Mge + dagger.(Mge))
        J = γ -> √γ .* Mde
    else
        Mgg = [embed(bN, i, transition(b1, 1, 1)) for i = 1:Na]
        Mee = [embed(bN, i, transition(b1, 2, 2)) for i = 1:Na]
        Mdd = [embed(bN, i, transition(b1, 3, 3)) for i = 1:Na]
        Mge = [embed(bN, i, transition(b1, 1, 2)) for i = 1:Na]
        Mde = [embed(bN, i, transition(b1, 3, 2)) for i = 1:Na]

        if interaction == :nn
            Vex = [Mge[i] * Mge[i + 1]' + Mge[i + 1] * Mge[i]' for i = 1:Na-1]
        else
            Vex = [
                sum([
                    abs(i - j)^(-3) * (Mge[i] * Mge[j]' + Mge[j] * Mge[i]')
                    for j = i+1:Na
                ])
                for i = 1:Na-1
            ]
        end

        H = (Ω, V) -> Ω / 2 * sum(Mge + dagger.(Mge)) + V * sum(Vex)
        J = γ -> √γ .* Mde
    end
    return model_me(Na, bN, ψ0, Mgg, Mee, Mdd, Mge, Mde, Vex, H, J)
end

"""
    analytical_spinwave_nh(Na, Ω, V, γ)

Build the single-excitation spin-wave effective Hamiltonian and initial state.
This model keeps `|G⟩` and spin-wave modes `|ℓ⟩` (`ℓ=1..Na`).
"""
function analytical_spinwave_nh(Na, Ω, V, γ)
    Δ = 2 * V * cos.(π / (1 + Na) * (1:Na)) .- im * γ / 2
    u = sin.(π / (1 + Na) * (1:Na) * (1:Na)')
    Ωj = Ω * sum(u, dims = 2) ./ sqrt.(sum(u .^ 2, dims = 2))
    H = diagm([0; Δ])
    H[1, 2:end] = Ωj ./ 2
    H[2:end, 1] = Ωj ./ 2
    ψ0 = [1; zeros(Na)]
    return H, ψ0
end

"""
    ground_single_sector_nh(Na, Ω, V, γ)

Build the strict `|G⟩ + {|e_j⟩}` (`j=1..Na`) effective non-Hermitian model
with long-range (`1/r^3`) exchange in the single-excitation manifold.
"""
function ground_single_sector_nh(Na, Ω, V, γ; sigma = 0.0)
    H = zeros(ComplexF64, Na + 1, Na + 1)
    x = collect(1:Na) .+ sigma * randn(Na)

    # Coupling between ground and local single excitations.
    for j = 1:Na
        H[1, j + 1] = Ω / 2
        H[j + 1, 1] = Ω / 2
        H[j + 1, j + 1] = -im * γ / 2
    end

    # Long-range exchange in the single-excitation manifold.
    for i = 1:Na-1
        for j = i+1:Na
            Jij = abs(x[i] - x[j])^(-3)
            H[i + 1, j + 1] += V * Jij
            H[j + 1, i + 1] += V * Jij
        end
    end

    ψ0 = zeros(ComplexF64, Na + 1)
    ψ0[1] = 1.0 + 0im
    return H, ψ0
end

"""
    ground_single_sector_nh_nn(Na, Ω, V, γ)

Build the strict `|G⟩ + {|e_j⟩}` model with nearest-neighbor exchange in the
single-excitation manifold. Set `sigma` for coupling noise.
"""
function ground_single_sector_nh_nn(Na, Ω, V, γ; sigma = 0.0)
    H = zeros(ComplexF64, Na + 1, Na + 1)

    for j = 1:Na
        H[1, j + 1] = Ω / 2
        H[j + 1, 1] = Ω / 2
        H[j + 1, j + 1] = -im * γ / 2
    end

    for j = 1:Na-1
        Jij = 1.0 + sigma * randn()
        H[j + 1, j + 2] += V * Jij
        H[j + 2, j + 1] += V * Jij
    end

    ψ0 = zeros(ComplexF64, Na + 1)
    ψ0[1] = 1.0 + 0im
    return H, ψ0
end

"""
    model_nh_sector01

Container for the strict zero+single-excitation non-Hermitian model.
"""
mutable struct model_nh_sector01
    Na::Int64
    ψ0::Vector{ComplexF64}
    H::Function
    sector::Symbol
end

"""
    model_nh_sector01(Na)

Build the strict `|G⟩ + {|e_j⟩}` reduced non-Hermitian model with long-range
exchange in the single-excitation manifold.
"""
function model_nh_sector01(Na; sigma = 0.0)
    ψ0 = zeros(ComplexF64, Na + 1)
    ψ0[1] = 1.0 + 0im
    H = (Ω, V, γ) -> first(ground_single_sector_nh(Na, Ω, V, γ; sigma = sigma))
    return model_nh_sector01(Na, ψ0, H, :zero_plus_one)
end

"""
    model_nh_sector01_nn(Na)

Build the strict `|G⟩ + {|e_j⟩}` reduced non-Hermitian model with nearest-
neighbor exchange in the single-excitation manifold.
"""
function model_nh_sector01_nn(Na; sigma = 0.0)
    ψ0 = zeros(ComplexF64, Na + 1)
    ψ0[1] = 1.0 + 0im
    H = (Ω, V, γ) -> first(ground_single_sector_nh_nn(Na, Ω, V, γ; sigma = sigma))
    return model_nh_sector01(Na, ψ0, H, :zero_plus_one)
end

"""
    ModelSpec

Categorized model specification.

Choices:
- `levels`: `:two_level` | `:three_level`
- `dynamics`: `:nonhermitian` | `:lindblad`
- `interaction`: `:nn` | `:C3`
- `noise`: `:clean` | `:noisy`
- `sector`: `:full_hilbert` | `:zero_plus_one`

Constraints:
- `levels = :three_level` requires `dynamics = :lindblad`.
- `dynamics = :lindblad` requires `levels = :three_level`.
- `levels = :three_level` supports only `noise = :clean`.
- `sector = :zero_plus_one` is only supported for
  `levels = :two_level` and `dynamics = :nonhermitian`.

`noise_strength` is used only when `noise = :noisy`.
"""
struct ModelSpec
    Na::Int64
    levels::Symbol
    dynamics::Symbol
    interaction::Symbol
    noise::Symbol
    sector::Symbol
    noise_strength::Float64
end

"""
    ModelSpec(; Na,
              levels=:two_level,
              dynamics=:nonhermitian,
              interaction=:nn,
              noise=:clean,
              sector=:full_hilbert,
              noise_strength=0.0)

Create a validated model specification.

Allowed keyword values:
- `levels`: `:two_level`, `:three_level`
- `dynamics`: `:nonhermitian`, `:lindblad`
- `interaction`: `:nn`, `:C3`
- `noise`: `:clean`, `:noisy`
- `sector`: `:full_hilbert`, `:zero_plus_one`
"""
function ModelSpec(; Na::Integer,
                   levels::Symbol = :two_level,
                   dynamics::Symbol = :nonhermitian,
                   interaction::Symbol = :nn,
                   noise::Symbol = :clean,
                   sector::Symbol = :full_hilbert,
                   noise_strength::Real = 0.0)
    Na < 1 && throw(ArgumentError("Na must be >= 1"))
    levels ∉ (:two_level, :three_level) && throw(ArgumentError("levels must be :two_level or :three_level"))
    dynamics ∉ (:nonhermitian, :lindblad) && throw(ArgumentError("dynamics must be :nonhermitian or :lindblad"))
    interaction ∉ (:nn, :C3) && throw(ArgumentError("interaction must be :nn or :C3"))
    noise ∉ (:clean, :noisy) && throw(ArgumentError("noise must be :clean or :noisy"))
    sector ∉ (:full_hilbert, :zero_plus_one) && throw(ArgumentError("sector must be :full_hilbert or :zero_plus_one"))

    if levels == :three_level && dynamics != :lindblad
        throw(ArgumentError("three-level models currently support only :lindblad dynamics"))
    end
    if dynamics == :lindblad && levels != :three_level
        throw(ArgumentError("lindblad models currently use the three-level model"))
    end
    if sector == :zero_plus_one && !(levels == :two_level && dynamics == :nonhermitian)
        throw(ArgumentError("sector=:zero_plus_one is currently supported only for two-level non-Hermitian models"))
    end
    if levels == :three_level && noise == :noisy
        throw(ArgumentError("three-level noisy model variant is not implemented"))
    end

    return ModelSpec(
        Int64(Na),
        levels,
        dynamics,
        interaction,
        noise,
        sector,
        Float64(noise_strength),
    )
end

"""
    build_model(spec::ModelSpec)

Build a model from categorized specification.
"""
function build_model(spec::ModelSpec)
    if spec.levels == :two_level && spec.dynamics == :nonhermitian
        σ = spec.noise == :clean ? 0.0 : spec.noise_strength
        if spec.sector == :zero_plus_one
            if spec.interaction == :nn
                return model_nh_sector01_nn(spec.Na; sigma = σ)
            else
                return model_nh_sector01(spec.Na; sigma = σ)
            end
        end

        if spec.interaction == :nn
            return model_nh_nn(spec.Na; sigma = σ)
        else
            return model_nh(spec.Na; sigma = σ)
        end
    end

    if spec.levels == :three_level && spec.dynamics == :lindblad
        return model_me(spec.Na; interaction = spec.interaction)
    end

    throw(ArgumentError("Unsupported model specification: $(spec)"))
end

"""
    build_model(; kwargs...)

Keyword constructor for categorized model creation.
"""
function build_model(; kwargs...)
    return build_model(ModelSpec(; kwargs...))
end
