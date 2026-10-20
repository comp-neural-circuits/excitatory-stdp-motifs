# ==============================================================================
# This code is used in the manuscript: 
# 'Heterogeneous plasticity rules generate diverse structures in recurrent networks'
# authors: Claudia Cusseddu, Dylan Festa, Christoph Miehl, Julijana Gjorgjieva
# affiliation: Technical University of Munich, Germany 
# contact: claudia.cusseddu@tum.de
# biorxiv preprint: https://www.biorxiv.org/content/10.1101/123456789 (2026)
# =============================================================================
using DrWatson
@quickactivate
using IterTools
using HawkesSimulator ; global const H = HawkesSimulator
using LinearAlgebra, Statistics
using ProgressMeter
using InvertedIndices # for Not()
using Printf
using JLD2
using Random

# ==============================================================================
# SCRIPT DESCRIPTION
# ==============================================================================

# This script runs a parameter sweep over a recurrent spiking network (Hawkes process model)
# with 75 excitatory and 25 inhibitory neurons. 
# A single Hebbian STDP plasticity rule (RULE = 'H', 'A', 'S') on E->E synapses, 
# and Vogels' inhibitory plasticity rule on I->E synapses.
#
# All excitatory neurons share the same target rate `he` (homogeneous
# population), and are recurrently connected. 
# Each excitatory neuron is paired to a single inhibitory neuron. 
#
# For every combination of (theta, gamma) in the grid below, `n_trials`
# independent repetitions are run (different random seeds), and results are
# saved to individual .jld2 files.
#
# To reload a result file later for analysis in Julia:
#   using JLD2
#   data = load("Hth1.00_ga1.00_tr1.jld2")
#   data["result"]     # -> full results Dict
#   data["mypars"]     # -> lightweight parameters Dict
#   data["total_spikes"]
# ==============================================================================

RULE = 'H'      # plasticity rule for E->E synapses: 'H' = Hebbian (PairSTDP). Change it to:
                # 'A' = anti-Hebbian (PairSTDP with swapped potentiation/depression),
                # 'S' = symmetric (SymmetricSTDP). Also used in output file naming. 
                # see lines 130-150 for how to change the plasticity rule in the code.

@assert RULE in ('H', 'A', 'S') "RULE must be one of 'H', 'A', 'S'"

# ------------------------------------------------------------------------------
# Parameter grid definition
# ------------------------------------------------------------------------------
# theta/gamma ranges depend on the plasticity rule: 'H' explores a broad range,
# while 'A' and 'S' explore a narrow range close to 1 (symmetric/balanced regime).
if RULE == 'H'
    const theta_range = [0.5, 0.9, 1.0, 1.5, 5.0] # depression/potentiation ratio parameter, 25 values
    const gamma_range = [1.1, 1.25] #[0.8, 1.0, 1.1, 1.25]  # ratio of STDP time constants (tau_minus/tau_plus), 17 values
else # RULE in ('A', 'S')
    const theta_range = [0.9, 1.2]  # depression/potentiation ratio parameter, 41 values
    const gamma_range = [0.9, 2.0]  # ratio of STDP time constants (tau_minus/tau_plus), 9 values
end
const n_trials = 10               # number of repeated trials per (theta, gamma) pair

# Full grid of (theta, gamma, trial_index) tuples, one entry per simulation to run
const grid_params = [(t, g, tr) for t in theta_range, g in gamma_range, tr in 1:n_trials]
const total_sims = length(grid_params) 

println("Launching ", total_sims, " new targeted simulations.")

"""
    mysim(cnt, total_spikes, Δt_rates, theta_val, gamma_val)
Run a single simulation of the recurrent E-I network for a given pair of
plasticity parameters `(theta_val, gamma_val)`, and return two dictionaries:

- `dict_tmp`   : full results (weights over time, spike trains, rates, etc.)
- `only_params`: lightweight dictionary with just the simulation metadata/parameters

Arguments:
- `cnt`          : integer used both as the RNG seed and as a simulation identifier
- `total_spikes` : total number of spikes to simulate (defines simulation length)
- `Δt_rates`     : time bin (s) used to compute instantaneous firing rates
- `theta_val`    : nominal value of the STDP depression/potentiation ratio parameter
- `gamma_val`    : nominal value of the STDP time-constant ratio parameter
"""
function mysim(cnt::Int64, total_spikes::Int64, Δt_rates::Float64, theta_val::Float64, gamma_val::Float64; w_start = nothing, t_begin = 0.0)
    Random.seed!(cnt)

    θ1 = theta_val
    γ1 = gamma_val

    # --------------------------------------------------------------------
    # Network size and structure
    # --------------------------------------------------------------------
    A1 = 1E-5              # STDP learning rate scale
    ne, ni = 75, 25           # number of excitatory / inhibitory neurons
    n = ne + ni
    idxe, idxi = 1:ne, ne .+ (1:ni)   # index ranges for E and I populations within the full weight matrix

    # Membrane/synaptic time constants and baseline external inputs
    τe, τi, he, hi = 10E-3, 10E-3, 10.0, 10.0

    # STDP / Vogels rule time constants and target rate
    tauplus, tauminus, τvogels, target_post = 20E-3, 20E-3, 40E-3, 5.0

    # Weight bounds for each connection type (E->E, E->I, I->E, I->I)
    w_min, w_max_ee, w_max_ei, w_max_ie, w_max_ii = 1E-5, 1.0, 10.0, 2.0, 1.0
    w_inh_ei = 5.0   # initial (fixed) diagonal E->I inhibitory weight

    # --------------------------------------------------------------------
    # Build the initial weight matrix w_mat (n x n), block by block.
    # Rows/columns 1:ne are excitatory, ne+1:n are inhibitory.
    # --------------------------------------------------------------------
    if isnothing(w_start)
        w_mat = zeros(n, n)
        w_mat[idxe, idxe] .= w_min .+ w_max_ee .* rand(ne, ne)
        w_mat[idxe, idxi] .= -w_inh_ei
        w_mat[idxi, idxe] .= w_max_ie
        w_mat[idxi, idxi] .= -w_max_ii
        w_mat[diagind(w_mat)] .= 0.0
    else
        # Load weights from a previous simulation (Part 2)
        @info "Resuming simulation: loading provided weights."
        w_mat = copy(w_start)
    end   
    w_0 = copy(w_mat)

    # --------------------------------------------------------------------
    # Build the vector of external input rates `h_vec`.
    # All excitatory neurons share the same target rate `he` (homogeneous
    # population), and all inhibitory neurons share `hi`.
    # --------------------------------------------------------------------
    h_vec = vcat(fill(he, ne), fill(hi, ni))

    # --------------------------------------------------------------------
    # Population states and synaptic traces.
    # A "trace" is required alongside each population state to support the
    # exponential-kernel connections used below.
    # --------------------------------------------------------------------
    pse, trae = H.population_state_exp_and_trace(ne, τe; label="population_e")
    psi, trai = H.population_state_exp_and_trace_inhibitory(ni, τi; label="population_i")

    # --------------------------------------------------------------------
    # E->E plasticity rule, selected by the global `RULE`:
    #   'H' (Hebbian)      : standard pair-based STDP, potentiation for pre->post
    #                        order (τplus), depression for post->pre order (τminus).
    #   'A' (anti-Hebbian) : same PairSTDP rule, but with potentiation/depression
    #                        (and their time constants) swapped.
    #   'S' (symmetric)    : SymmetricSTDP rule, same sign for both pre->post and
    #                        post->pre pairings.
    # In all three cases Aplus/Aminus/τminus are derived the same way from
    # A1, tauplus, and the theta/gamma parameters.
    # --------------------------------------------------------------------
    plasticity_ee = let A = A1,
        τplus = tauplus,
        γ = γ1,
        θ = θ1,
        Aplus = A / τplus,
        τminus = γ * τplus,
        Aminus = -θ * A / τminus
        bounds = H.PlasticityBoundsLowHigh(w_min, w_max_ee)
        if RULE == 'H'
            H.PairSTDP(τplus, τminus, Aplus, Aminus, ne, ne; plasticity_bounds=bounds)
        elseif RULE == 'A'
            H.PairSTDP(τminus, τplus, Aminus, Aplus, ne, ne; plasticity_bounds=bounds)
        else # RULE == 'S'
            H.SymmetricSTDP(τplus, τminus, Aplus, Aminus, ne, ne; plasticity_bounds=bounds)
        end
    end

    # --------------------------------------------------------------------
    # I->E plasticity: Vogels' inhibitory plasticity rule.
    # Regulates E->I feedback so that excitatory neurons approach a target rate.
    # Note: amplitude `A` here is NOT divided by tau, hence the larger value.
    # --------------------------------------------------------------------
    plasticity_ei = let A = 5E-5,
        tau = τvogels,
        target_rate = target_post,
        bounds = H.PlasticityBoundsLowHigh(w_min, w_max_ei)
        H.PlasticityInhibitory(tau, A, ne, ni; r_target=target_rate, plasticity_bounds=bounds)
    end

    # --------------------------------------------------------------------
    # Build the four synaptic connections (with exponential kernels).
    # Only E->E and I->E carry a plasticity rule; I->E and I->I are static
    # (note: I->E here is fixed/static too, only E->I is plastic via Vogels).
    # --------------------------------------------------------------------
    connection_ee = H.ConnectionExpKernel(w_mat[idxe, idxe], trae, plasticity_ee)
    connection_ie = H.ConnectionExpKernel((w_mat[idxi, idxe]), trae)
    connection_ei = H.ConnectionExpKernel(abs.(w_mat[idxe, idxi]), trai, plasticity_ei)
    connection_ii = H.ConnectionExpKernel(abs.(w_mat[idxi, idxi]), trai)

    # Assemble the two populations, each receiving recurrent input from itself
    # and from the other population.
    pop_e = H.PopulationExpKernel(pse, h_vec[idxe], (connection_ee, pse), (connection_ei, psi))
    pop_i = H.PopulationExpKernel(psi, h_vec[idxi], (connection_ie, pse), (connection_ii, psi))

    # ======================================================================
    # RECORDERS
    # ======================================================================
    # We record: (1) full spike trains for both populations, and
    # (2) the four weight matrices, sampled every Δt_wrec seconds up to Tend_wrec.
    n_spikes = total_spikes
    rec_spikes_ei = H.RecFullTrain(n_spikes + 10, 2)   # "2" = number of populations (E and I)

    Δt_wrec = 250.0       # weight-recording interval (s)
    Tend_wrec = 750_000.0   # expected simulation duration (s) used to size the weight recorder
    # (this is just a guess for pre-allocation purposes; actual runtime is determined by n_spikes)

    rec_wee = H.RecTheseWeights(connection_ee.weights, Δt_wrec, Tend_wrec)
    rec_wie = H.RecTheseWeights(connection_ie.weights, Δt_wrec, Tend_wrec)
    rec_wei = H.RecTheseWeights(connection_ei.weights, Δt_wrec, Tend_wrec)
    rec_wii = H.RecTheseWeights(connection_ii.weights, Δt_wrec, Tend_wrec)

    # Build the full network object: populations + all recorders together.
    the_network = H.RecurrentNetworkExpKernel(
        (pop_e, pop_i),
        (rec_spikes_ei, rec_wee, rec_wie, rec_wei, rec_wii)
    )

    """
        run_simulation!(network, num_spikes; show_progress=true, t_start=0.0)

    Advance the network dynamics event-by-event (one spike per iteration) for
    `num_spikes` steps, starting from time `t_start`. Returns the final time.
    """
    function run_simulation!(network, num_spikes; show_progress=true, t_start::Float64=0.0)
        t_now = t_start
        # H.reset!(network) # would clear spike trains etc. if a fresh restart were needed
        pr = Progress(n_spikes; enabled=show_progress)
        for _ in 1:num_spikes
            t_now = H.dynamics_step!(t_now, network)
            ProgressMeter.next!(pr)
        end
        ProgressMeter.finish!(pr)
        return t_now
    end

    # --------------------------------------------------------------------
    # Run the simulation
    # --------------------------------------------------------------------
    t_end = run_simulation!(the_network, n_spikes; show_progress=true, t_start = t_begin)
    println("Simulation completed at time $(round(t_end/60.;digits=1)) min or $(round(t_end/3600.;digits=1)) hours")

    # ======================================================================
    # POST-PROCESSING: extract data from the recorders
    # ======================================================================
    weights_contentee = H.get_content(rec_wee)
    weights_content_ie = H.get_content(rec_wie)
    weights_content_ei = H.get_content(rec_wei)
    weights_content_ii = H.get_content(rec_wii)
    spikes_content = H.get_content(rec_spikes_ei)

    """
        assemble_weight(wee, wie, wei, wii)

    Reassemble the four recorded weight sub-matrices into a single full
    n x n weight matrix, restoring the correct sign convention for the
    inhibitory blocks (E->I and I->I are stored as positive magnitudes and
    must be negated here).
    """
    function assemble_weight(wee::M, wie::M, wei::M, wii::M) where M<:AbstractMatrix
        w_mat = zeros(Float64, n, n)
        w_mat[idxe, idxe] = wee
        w_mat[idxi, idxe] = wie
        w_mat[idxe, idxi] = -wei
        w_mat[idxi, idxi] = -wii
        return w_mat
    end

    times_w = weights_contentee.times
    # Reconstruct the full weight matrix at every recorded time point
    weights = let k = length(weights_contentee.weights)
        map(i -> assemble_weight(weights_contentee.weights[i],
                                  weights_content_ie.weights[i],
                                  weights_content_ei.weights[i],
                                  weights_content_ii.weights[i]), 1:k)
    end

    """
        get_spike_window(trains, t_start, t_end)

    For each neuron's spike train in `trains`, return only the spike times
    falling within the half-open interval [t_start, t_end).
    """
    function get_spike_window(trains, t_start, t_end)
        return [filter(t -> t_start <= t < t_end, neuron_train) for neuron_train in trains]
    end

    trains_e = H.get_trains(spikes_content; Nneus=ne, pop_idx=1)
    trains_i = H.get_trains(spikes_content; Nneus=ni, pop_idx=2)

    t_mid = t_end / 2

    # Save short (600 s / 600 s around midpoint) snapshots of spiking activity
    # at the beginning, middle, and end of the simulation, for later inspection.
    spikes_initial = Dict(
        "e" => get_spike_window(trains_e, 0.0, 600.0),
        "i" => get_spike_window(trains_i, 0.0, 600.0)
    )

    spikes_mid = Dict(
        "e" => get_spike_window(trains_e, t_mid - 300.0, t_mid + 300.0),
        "i" => get_spike_window(trains_i, t_mid - 300.0, t_mid + 300.0)
    )

    spikes_final = Dict(
        "e" => get_spike_window(trains_e, t_end - 600.0, t_end),
        "i" => get_spike_window(trains_i, t_end - 600.0, t_end)
    )

    # Instantaneous population firing rates, binned every Δt_rates seconds
    times_r, rates_insta_i = H.instantaneous_rates(trains_i, Δt_rates, t_end)
    times_r, rates_insta_e = H.instantaneous_rates(trains_e, Δt_rates, t_end)
    sim_rates = Dict("sim_rates_e" => rates_insta_e, "sim_rates_i" => rates_insta_i)
    sim_spikes = Dict("initial" => spikes_initial, "mid" => spikes_mid, "final" => spikes_final)

    sim_wfin = copy(weights[end])   # final weight matrix
    taus = Dict("tauplus" => tauplus, "tauminus" => tauminus, "taue" => τe, "taui" => τi, "tauvogels" => τvogels)

    # Mean off-diagonal weight, over the whole network and restricted to E->E only,
    # tracked across all recorded time points (useful to check convergence/stability)
    mean_w_notdiag = [mean(W[Not(diagind(W))]) for W in weights]
    mean_w_ee_notdiag = [mean(W[idxe, idxe][Not(diagind(W[idxe, idxe]))]) for W in weights]

    # Full results dictionary (everything needed for later analysis/plotting)
    dict_tmp = Dict(
        "myseed" => cnt, "h_vec" => h_vec, "A1" => A1, "theta1" => θ1, "gamma1" => γ1,
        "t_end" => t_end, "sim_w0" => w_0, "sim_wfin" => sim_wfin, "sim_rates" => sim_rates,
        "sim_spikes" => sim_spikes, "mean_w" => mean_w_notdiag, "mean_w_ee" => mean_w_ee_notdiag,
        "taus" => taus, "targetrate" => target_post, "he" => he, "sim_tend" => t_end,
        "weights" => weights, "weights_times" => times_w
    )

    # Lightweight dictionary with only the simulation parameters/metadata
    only_params = Dict(
        "myseed" => cnt, "A1" => A1, "theta1" => θ1, "gamma1" => γ1, "t_end" => t_end,
        "taus" => taus, "targetrate" => target_post, "he" => he, "hi" => hi
    )

    return dict_tmp, only_params
end

# ==============================================================================
# PARALLEL EXECUTION OVER THE PARAMETER GRID
# ==============================================================================

const nthreads = Threads.nthreads()
@info "Using $nthreads threads"

# Thread-safe counter (atomic) used to track how many simulations remain
mutable struct Atomic{T}; @atomic x::T; end

nspikes = 100_000_000      # number of spikes per simulation (defines its duration)
delta_t_rates = 250.0     # bin size (s) for instantaneous rate computation

"""
    time_me_please()
Run all simulations in `grid_params` in parallel across available threads.
Results are saved under `datadir("sims", "rule_\$RULE")`, one .jld2 file per
(theta, gamma, trial) combination. For each combination:
  - build the expected output filename
  - skip the simulation if that file already exists (allows resuming a sweep)
  - otherwise run `mysim` and save the results, together with rule/trial/theta/gamma metadata
"""
function time_me_please()
    # All files for this RULE go into their own subfolder, e.g. data/sims/rule_H/
    sims_dir = datadir("sims_networks", "rule_$RULE")
    mkpath(sims_dir)
    sim_counter = Atomic(total_sims)
    Threads.@threads for k in 1:total_sims
        θ_val, γ_val, trial_idx = grid_params[k]
        file_name = joinpath(sims_dir, @sprintf("%cnet_th%.2f_ga%.2f_tr%d.jld2", RULE, θ_val, γ_val, trial_idx))
        # Skip already-completed simulations (enables resuming an interrupted sweep)
        if isfile(file_name)
            @atomic sim_counter.x -= 1
            continue
        end
        this_seed = 1234 + k
        result, mypars = mysim(this_seed, nspikes, delta_t_rates, θ_val, γ_val)
        @atomic sim_counter.x -= 1
        @info "Completed: θ=$(round(θ_val, digits=2)), γ=$(round(γ_val, digits=2)), Trial=$trial_idx. Remaining: $(sim_counter.x)"
        output = Dict("result" => result, "mypars" => mypars, "total_spikes" => nspikes,
                       "rule" => RULE, "trial" => trial_idx, "theta_nominal" => θ_val, "gamma_nominal" => γ_val)
        save(file_name, output)
    end
    return nothing
end
# dry run
mysim(1, 1000, 1.0, 1., 1.)
@time time_me_please()