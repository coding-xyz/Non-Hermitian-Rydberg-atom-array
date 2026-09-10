"""
    eigen_nh(H)

Compute eigenvalues and biorthogonal right/left eigenvectors of a non-Hermitian
operator or matrix.
"""
function eigen_nh(H)
    if typeof(H) <: Operator
        H = dense(H).data
    end
    lambda_u, R = eigen(H)
    _, L = eigen(H')
    ml, mu = lu(R * L', NoPivot())
    return lambda_u, inv(ml) * R, inv(mu)' * L
end

const _EXPORT_ALIASES = Dict{Symbol, Symbol}(
    :evals => :eigenvalues,
    :eigenvalues => :eigenvalues,
    :fidelity => :fidelity,
    :populations => :populations,
    :wavefunction => :wavefunction,
)

function _normalize_exports(exports)
    ex = exports isa Symbol ? (exports,) : Tuple(exports)
    isempty(ex) && throw(ArgumentError("exports cannot be empty"))
    norm = Symbol[]
    for s in ex
        s isa Symbol || throw(ArgumentError("exports entries must be Symbols, got $(typeof(s))"))
        push!(norm, get(_EXPORT_ALIASES, s, s))
    end
    return Tuple(norm)
end

function _validate_exports(exports, allowed::Tuple, fname::AbstractString)
    bad = [s for s in exports if !(s in allowed)]
    isempty(bad) || throw(ArgumentError("Unsupported exports in $(fname): $(bad). Allowed: $(allowed)"))
end

function _package_result(; t, exports, fidelity = nothing, eigenvalues = nothing, populations = nothing, wavefunction = nothing, lognorm = nothing)
    values = Dict{Symbol, Any}(
        :fidelity => fidelity,
        :eigenvalues => eigenvalues,
        :populations => populations,
        :wavefunction => wavefunction,
        :lognorm => lognorm,
    )
    out = Dict{Symbol, Any}(:t => t)
    for key in exports
        out[key] = values[key]
    end
    return out
end

_psi0(m::model_nh) = getfield(m, 3)
_psi0(m::model_me) = getfield(m, 3)
_psi0(m::model_nh_sector01) = getfield(m, 2)

function auto_time_grid_nh_estimate(Ω, V, γ;
                                    tmax_cap = 200.0,
                                    min_tmax = 10.0,
                                    min_dt = 0.02,
                                    max_dt = 0.5,
                                    tmax_factor = 8.0,
                                    dt_factor = 0.1)

    vals = [abs(x) for x in (Ω, V, γ) if abs(x) > 1e-12]

    if isempty(vals)
        tmax = min_tmax
        dt = max_dt
    else
        fast_scale = maximum(vals)
        slow_scale = minimum(vals)

        dt = clamp(dt_factor / fast_scale, min_dt, max_dt)
        tmax = clamp(tmax_factor / slow_scale, min_tmax, tmax_cap)
    end

    return collect(0.0:dt:tmax)
end


"""
    dynamics_nh(t, model::model_nh, Omega, V, gamma; exports=...)

Evolve the non-Hermitian model at times `t`.

Keyword options:
- `exports`: quantities to export (default `(:fidelity, :eigenvalues, :populations)`).
  Allowed: `:fidelity`, `:eigenvalues` (alias `:evals`), `:populations`, `:wavefunction`.

Returns a dictionary with key `:t` plus the requested export keys.
"""
function dynamics_nh(t, model::model_nh, Omega, V, gamma;
                     exports = (:fidelity, :eigenvalues, :populations))
    return dynamics_nh(
        t, dense(model.H(Omega, V, gamma)).data, _psi0(model).data;
        exports = exports,
    )
end

"""
    dynamics_nh(t, model::model_nh_sector01, Omega, V, gamma; exports=...)

Evolve the strict zero+single-excitation non-Hermitian model.
Returns a dictionary with key `:t` plus the requested export keys.
"""
function dynamics_nh(t, model::model_nh_sector01, Omega, V, gamma;
                     exports = (:fidelity, :eigenvalues, :populations))
    return dynamics_nh(
        t, model.H(Omega, V, gamma), _psi0(model);
        exports = exports,
    )
end

"""
    dynamics_nh(t, H, psi0; exports=...)

Evolve with matrix `H` and initial state `psi0`.
Returns a dictionary with key `:t` plus the requested export keys.
"""
function dynamics_nh(t, H, psi0;
                     exports = (:fidelity, :eigenvalues, :populations, :lognorm),
                     atol = 1e-12)
    exports = _normalize_exports(exports)
    _validate_exports(exports, (:fidelity, :eigenvalues, :populations, :lognorm, :wavefunction), "dynamics_nh")

    want_fidelity = :fidelity in exports
    want_populations = :populations in exports
    want_wavefunction = :wavefunction in exports
    want_evals = :eigenvalues in exports

    fidelity = want_fidelity ? zeros(Float64, length(t)) : nothing
    populations = want_populations ? zeros(length(psi0), length(t)) : nothing
    wavefunction = want_wavefunction ? zeros(ComplexF64, length(psi0), length(t)) : nothing

    eval_f, evec_f = eigen(H)
    U1 = evec_f
    lognorm = zeros(Float64, length(t))

    # 在本征基里存“已归一化”的系数
    a = U1 \ psi0
    psi = U1 * a
    n0 = norm(psi)
    n0 > atol || error("initial state has zero (or tiny) norm")

    a ./= n0
    psi ./= n0
    lognorm[1] = log(n0)

    if want_fidelity
        fidelity[1] = abs(dot(psi, psi0 / norm(psi0)))^2
    end
    if want_populations
        populations[:, 1] = abs2.(psi)
    end
    if want_wavefunction
        wavefunction[:, 1] = psi
    end

    for k in 2:length(t)
        dt = t[k] - t[k - 1]
        a .*= exp.(-im * dt .* eval_f)
        psi = U1 * a

        nk = norm(psi)
        if !isfinite(nk) || nk ≤ atol
            error("wavefunction norm became non-finite or too small at step $k, t=$(t[k])")
        end

        a ./= nk
        psi ./= nk
        lognorm[k] = lognorm[k - 1] + log(nk)

        if want_fidelity
            fidelity[k] = abs(dot(psi, psi0 / norm(psi0)))^2
        end
        if want_populations
            populations[:, k] = abs2.(psi)
        end
        if want_wavefunction
            wavefunction[:, k] = psi
        end
    end


    return _package_result(
        t = t,
        exports = exports,
        fidelity = fidelity,
        eigenvalues = want_evals ? eval_f : nothing,
        populations = populations,
        wavefunction = wavefunction,
        lognorm = lognorm
    )
end

"""
    dynamics_me(t, model::model_me, Omega, V, gamma; exports=...)

Solve the master equation.

Keyword options:
- `exports`: quantities to export (default `(:fidelity, :populations)`).
  Allowed: `:fidelity`, `:populations`.

Returns a dictionary with key `:t` plus the requested export keys.
"""
function dynamics_me(t, model::model_me, Omega, V, gamma;
                     exports = (:fidelity, :populations))
    exports = _normalize_exports(exports)
    _validate_exports(exports, (:fidelity, :populations), "dynamics_me")

    want_fidelity = :fidelity in exports
    want_populations = :populations in exports

    t, rho_t = timeevolution.master(t, _psi0(model), model.H(Omega, V), model.J(gamma))
    fidelity = want_fidelity ? [abs(rho_t[k].data[1, 1]) for k in axes(t, 1)] : nothing
    populations = want_populations ? [real(expect(model.Mgg[i], rho_t[k])) for i in 1:model.Na, k in axes(t, 1)] : nothing

    return _package_result(
        t = t,
        exports = exports,
        fidelity = fidelity,
        populations = populations,
    )
end

"""
    dynamics_ex(Na, t, Omega, V, gamma; exports=...)

Evolve the single-excitation reduced model.
Returns a dictionary with key `:t` plus the requested export keys.
"""
function dynamics_ex(Na, t, Omega, V, gamma;
                     exports = (:fidelity, :eigenvalues, :populations))
    H, psi0 = analytical_spinwave_nh(Na, Omega, V, gamma)
    return dynamics_nh(t, H, psi0; exports = exports)
end

"""
    dynamics_ground_single_sector(Na, t, Omega, V, gamma; exports=...)

Evolve the strict zero+single-excitation effective model.
Returns a dictionary with key `:t` plus the requested export keys.
"""
function dynamics_ground_single_sector(Na, t, Omega, V, gamma;
                                       exports = (:fidelity, :eigenvalues, :populations))
    H, psi0 = ground_single_sector_nh(Na, Omega, V, gamma)
    return dynamics_nh(t, H, psi0; exports = exports)
end

"""
    dynamics_me_trajectory(t, model::model_me, Omega, V, gamma)

Solve master-equation dynamics and return full density-matrix trajectory.
Returns a dictionary with keys `:t` and `:density`.
"""
function dynamics_me_trajectory(t, model::model_me, Omega, V, gamma)
    t_out, rho_t = timeevolution.master(t, model.ψ0, model.H(Omega, V), model.J(gamma))
    return Dict{Symbol, Any}(
        :t => t_out,
        :density => rho_t,
    )
end

"""
    dynamics_me_nojump_trajectory(t, model::model_me, Omega, V, gamma)

Solve conditional no-jump dynamics using the effective non-Hermitian Hamiltonian
`H_eff = H - i/2 * sum(J_k' * J_k)`.
Returns a dictionary with keys `:t` and `:wavefunction`.
"""
function dynamics_me_nojump_trajectory(t, model::model_me, Omega, V, gamma)
    J = model.J(gamma)
    Heff = model.H(Omega, V) - 0.5im * sum([Jk' * Jk for Jk in J])
    t_out, psi_t = timeevolution.schroedinger(t, model.ψ0, Heff)
    return Dict{Symbol, Any}(
        :t => t_out,
        :wavefunction => psi_t,
    )
end

"""
    dynamics_mcwf(t, model::model_me, Omega, V, gamma; trajectories=1000, exports=...)

Monte Carlo wave-function simulation for the three-level master-equation model.

Returns a Dict with:
- `:t`
- `:fidelity`: trajectory-averaged initial-state population
- `:populations`: trajectory-averaged site-resolved ground-state populations
- `:fidelity_trajectories`: one LE trace per MCWF trajectory, for Fig. S7 thin lines
"""
function dynamics_mcwf(t, model::model_me, Omega, V, gamma;
                       trajectories::Integer = 1000,
                       exports = (:fidelity, :populations),
                       seed::Integer = 1234,
                       threaded::Bool = true,
                       kwargs...)
    exports = exports isa Symbol ? (exports,) : Tuple(exports)

    allowed = (:fidelity, :populations)
    bad = [x for x in exports if !(x in allowed)]
    isempty(bad) || throw(ArgumentError("Unsupported exports in dynamics_mcwf: $(bad). Allowed: $(allowed)"))

    want_fidelity = :fidelity in exports
    want_populations = :populations in exports

    nt = length(t)
    ntraj = Int(trajectories)

    fidelity_traj = zeros(Float64, ntraj, nt)
    population_traj = want_populations ? zeros(Float64, model.Na, nt, ntraj) : nothing

    H = model.H(Omega, V)
    J = model.J(gamma)

    function run_one!(itraj)
        _, psi_t = timeevolution.mcwf(
            t, model.ψ0, H, J;
            seed = UInt(seed + itraj - 1),
            kwargs...
        )

        for k in eachindex(t)
            psi = psi_t[k]
            fidelity_traj[itraj, k] = abs2(dot(model.ψ0.data, psi.data))

            if want_populations
                for i in 1:model.Na
                    population_traj[i, k, itraj] = real(expect(model.Mgg[i], psi))
                end
            end
        end
    end

    if threaded
        Threads.@threads for itraj in 1:ntraj
            run_one!(itraj)
        end
    else
        for itraj in 1:ntraj
            run_one!(itraj)
        end
    end

    out = Dict{Symbol, Any}(:t => t)
    out[:fidelity_trajectories] = fidelity_traj

    if want_fidelity
        out[:fidelity] = vec(mean(fidelity_traj, dims = 1))
    end
    if want_populations
        out[:populations] = dropdims(mean(population_traj, dims = 3), dims = 3)
    end

    return out
end