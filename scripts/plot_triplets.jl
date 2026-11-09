# ==============================================================================
# This code is used in the manuscript: 
# 'Heterogeneous plasticity rules generate diverse structures in recurrent networks'
# authors: Claudia Cusseddu, Dylan Festa, Christoph Miehl, Julijana Gjorgjieva
# affiliation: Technical University of Munich, Germany 
# contact: claudia.cusseddu@tum.de
# biorxiv preprint: https://doi.org/10.64898/2026.10.26.760407 (2026)
# =============================================================================
using DrWatson
@quickactivate
using JLD2
using DataFrames
using LinearAlgebra
using Colors
using Plots
using Printf
using XLSX

# ==============================================================================
# ANALYSIS OF 3-NODE CONNECTIVITY MOTIFS FROM SIMULATION SWEEPS
# ==============================================================================
# For a given plasticity rule ('H', 'A', or 'S'), this script:
#   1. loads all simulation results saved by the simulation script under
#      datadir("sims", "rule_<RULE>")
#   2. classifies the final E->E weight matrix of every trial into one of
#      16 possible 3-node motifs, then groups those into 8 broader classes
#   3. builds a (theta, gamma) heatmap of the average motif-class color
# ==============================================================================
rule = 'H'  # 'H', 'A', or 'S'
const NE = 3          # number of excitatory neurons (must match the simulation script)
const W_THRESH = 0.12  # threshold above which a synapse is considered "connected"
const WMAX = 0.5      # E->E upper weight bound (must match `w_max_ee` in sim_triplets.jl)

# ------------------------------------------------------------------------------
# Motif class colors and labels (for the average-color heatmap)
# ------------------------------------------------------------------------------
const MOTIF_COLORMAP = [
    RGB(0.8353, 0.2431, 0.3098),  # 1: red        (Motif 1)
    RGB(0.9900, 0.6800, 0.3800),  # 2: orange     (Motif 2,4,5,6)
    RGB(0.9020, 0.9608, 0.5961),  # 3: yellow     (Motif 10,11)
    RGB(0.4000, 0.7608, 0.6471),  # 4: green      (Motif 12,13,14,15)
    RGB(0.7294, 0.3333, 0.8227),  # 5: pink       (Motif 3,9)
    RGB(0.1961, 0.5333, 0.7412),  # 6: blue       (Motif 16, all >= wmax)
    RGB(0.3922, 0.1961, 0.5882),  # 7: purple     (Motif 16, within bounds)
    RGB(1.0000, 1.0000, 1.0000),  # 8: white      (Motif 7,8)
]
const MOTIF_LABELS = [
    "1: unconnected",
    "2,4,5,6: no bidirectional",
    "10,11: unidirectional",
    "12-15: bidirectional",
    "3,9: symmetry",
    "16: all-to-all (>= wmax)",
    "16: within bounds",
    "7,8: others",
]

function motif_legend()
    for (i, label) in enumerate(MOTIF_LABELS)
        println(i, ": ", label)
    end
end

# ==============================================================================
# MOTIF CLASSIFICATION
# ==============================================================================
"""
    motif_distribution(W, wmax, w_thresh) -> Vector{Int}

Count the occurrences of each of the 16 canonical 3-node motifs in the
thresholded connectivity matrix `W` (entries > `w_thresh` are considered
connected). Returns a vector of 16 counts; for a 3x3 matrix exactly one
entry equals 1. (`wmax` is not used here; it is kept for a uniform signature.)
Based on the method of the following article:
S. Itzkovitz, R. Milo, N. Kashtan, G. Ziv, and U. Alon 
"Subgraphs in random networks", Phys Review E 68, 026127 (2003)
https://journals.aps.org/pre/pdf/10.1103/PhysRevE.68.026127
"""
function motif_distribution(W::AbstractMatrix{<:Real}, wmax::Real, w_thresh::Real)
    N = size(W, 1)

    M = Float64.(W .> w_thresh)
    M = M - diagm(0 => diag(M))
    M = M'
    S = M .* (M .== M')
    A = M - S
    Mtilde = ones(N, N) - M
    Stilde = Mtilde .* (Mtilde .== Mtilde')
    Stilde = Stilde - diagm(0 => diag(Stilde))

    m1  = round(Int, sum((Stilde * Stilde) .* Stilde) / 6)
    m2  = round(Int, sum((Stilde * Stilde) .* A))
    m3  = round(Int, sum((Stilde * Stilde) .* S) / 2)
    m4  = round(Int, (sum((A' * A) .* Mtilde .* Mtilde') - tr(A' * A)) / 2)
    m5  = round(Int, (sum((A * A') .* Mtilde .* Mtilde') - tr(A * A')) / 2)
    m6  = round(Int, sum((A * A) .* Mtilde .* Mtilde'))
    m7  = round(Int, sum((S * A') .* Mtilde .* Mtilde'))
    m8  = round(Int, sum((S * A) .* Mtilde .* Mtilde'))
    m9  = round(Int, (sum((S * S) .* Mtilde .* Mtilde') - tr(S * S)) / 2)
    m10 = round(Int, sum((A * A) .* A))
    m11 = round(Int, sum((A' * A') .* A) / 3)
    m12 = round(Int, sum((A' * A) .* S) / 2)
    m13 = round(Int, sum((A * A) .* S))
    m14 = round(Int, sum((A * A') .* S) / 2)
    m15 = round(Int, sum((S * S) .* A))
    m16 = round(Int, sum((S * S) .* S) / 6)

    motif_counts = [
    m1, m2, m3, m4, m5, m6, m7, m8,
    m9, m10, m11, m12, m13, m14, m15, m16
    ]

    return motif_counts
end

"""
    classify_motif_3node(W, wmax, w_thresh) -> (motif_idx, class_id)

Classify a 3x3 connectivity matrix `W` into one of the 16 canonical 3-node
motifs (`motif_idx`, see `motif_distribution`) and map it to one of the 8
broader classes (`class_id`, see `MOTIF_LABELS`). Motif 16 is split into
class 6 (all off-diagonal weights >= `wmax`) and class 7 (within bounds).
Returns `(nothing, missing)` if no motif matches (should not normally happen).
"""
function classify_motif_3node(W::AbstractMatrix{<:Real}, wmax::Real, w_thresh::Real)
    N = size(W, 1)
    # warning that the size must be 3x3 (because we find the motif index whose count is 1)
    N != 3 && error("Input weight matrix must be 3x3, but got size $(size(W))")

    motif_counts = motif_distribution(W, wmax, w_thresh)
    motif_idx = findfirst(==(1), motif_counts)
    motif_idx === nothing && return nothing, missing

    class_id = if motif_idx == 1
        1
    elseif motif_idx in (2, 4, 5, 6)
        2
    elseif motif_idx in (10, 11)
        3
    elseif motif_idx in (12, 13, 14, 15)
        4
    elseif motif_idx in (3, 9)
        5
    elseif motif_idx == 16
        off_diag_W = [W[i, j] for i in 1:N, j in 1:N if i != j]
        all(off_diag_W .>= 0.99 * wmax) ? 6 : 7
    elseif motif_idx in (7, 8)
        8
    else
        missing
    end

    return motif_idx, class_id
end

# ==============================================================================
# LOADING RESULTS INTO A DATAFRAME
# ==============================================================================
"""
    load_results_table(rule; sims_dir=datadir("sims", "rule_\$rule")) -> DataFrame

Load every .jld2 result file for a given plasticity `rule` ('H', 'A', or 'S')
and assemble a DataFrame with one row per (theta, gamma, trial) simulation.

Columns: `theta`, `gamma`, `trial`, `seed`, `t_end`, `final_weights` (the 3x3
E->E block of the final weight matrix).
"""
function load_results_table(rule::Char; sims_dir::String=datadir("sims", "rule_$rule"))
    isdir(sims_dir) || error("Directory not found: $sims_dir")
    files = filter(f -> endswith(f, ".jld2"), readdir(sims_dir; join=true))
    isempty(files) && error("No .jld2 files found in $sims_dir")
    println("Found $(length(files)) simulation files.")

    rows = NamedTuple[]
    for f in files
        d = load(f)
        result = d["result"]
        Wfull = result["sim_wfin"]
        push!(rows, (
            theta = d["theta_nominal"],
            gamma = d["gamma_nominal"],
            trial = d["trial"],
            seed = result["myseed"],
            t_end = result["t_end"],
            final_weights = Wfull[1:NE, 1:NE],
        ))
    end
    return DataFrame(rows)
end

# ==============================================================================
# BUILDING THE (theta, gamma) AVERAGE-COLOR HEATMAP
# ==============================================================================
"""
    compute_motif_heatmap(df; w_thresh=W_THRESH, wmax=WMAX)

Classify every trial's final weight matrix and build the average-color
heatmap: a (n_gamma, n_theta, 3) array giving, at each (theta, gamma) point,
the average RGB color of the motif classes observed across trials.

Returns a NamedTuple `(unique_thetas, unique_gammas, heatmap_avg, df)` where
`df` is a copy of the input with two new columns, `motif_id` and `class_id`,
filled in per row.
"""
function compute_motif_heatmap(df::DataFrame; w_thresh::Real=W_THRESH, wmax::Real=WMAX)
    df = copy(df)
    df.motif_id = Vector{Union{Int,Nothing}}(nothing, nrow(df))
    df.class_id = Vector{Union{Int,Missing}}(missing, nrow(df))

    unique_thetas = sort(unique(df.theta))
    unique_gammas = sort(unique(df.gamma))

    heatmap_avg = zeros(length(unique_gammas), length(unique_thetas), 3)

    for (i, th) in enumerate(unique_thetas), (j, ga) in enumerate(unique_gammas)
        rows = findall((df.theta .== th) .& (df.gamma .== ga))
        isempty(rows) && continue

        rgb_sum = zeros(3)
        for r in rows
            W = df.final_weights[r]
            W_filtered = W .* (W .> w_thresh)
            motif_idx, class_id = classify_motif_3node(W_filtered, wmax, w_thresh)
            df.motif_id[r] = motif_idx
            df.class_id[r] = class_id
            if !ismissing(class_id)
                c = MOTIF_COLORMAP[class_id]
                rgb_sum .+= (red(c), green(c), blue(c))
            end
        end

        heatmap_avg[j, i, :] .= rgb_sum ./ length(rows)   # matches MATLAB: divides by n trials, not n classified
    end

    return (; unique_thetas, unique_gammas, heatmap_avg, df)
end

# ------------------------------------------------------------------------------
# Helper: pick a readable subset of ticks (avoids overcrowding the axis when
# there are dozens of theta/gamma values, e.g. the 41-point 'A'/'S' grids)
# ------------------------------------------------------------------------------
function _sparse_ticks(values; max_ticks::Int=10)
    n = length(values)
    step = max(1, ceil(Int, n / max_ticks))
    idx = 1:step:n
    return (collect(idx), [@sprintf("%.2f", v) for v in values[idx]])
end

# ==============================================================================
# PLOTTING
# ==============================================================================
"""
    plot_motif_heatmap(res; title_str="motifs - average")

Plot the average-color motif heatmap (RGB composite) from the output of
`compute_motif_heatmap`. Call `motif_legend()` separately to interpret colors.
"""
function plot_motif_heatmap(res; title_str::String="motifs - average")
    img = [RGB(res.heatmap_avg[j, i, 1], res.heatmap_avg[j, i, 2], res.heatmap_avg[j, i, 3])
           for j in 1:length(res.unique_gammas), i in 1:length(res.unique_thetas)]
    xt = _sparse_ticks(res.unique_thetas)
    yt = _sparse_ticks(res.unique_gammas)
    plt = Plots.plot(img; yflip=false, xlabel="theta", ylabel="gamma", title=title_str,
                 xticks=xt, yticks=yt, size=(700, 600),
                 aspect_ratio=:none, grid=false,
                     xlims=(0.5, length(res.unique_thetas) + 0.5),
                     ylims=(0.5, length(res.unique_gammas) + 0.5))
    return plt
end


"""
    export_results_xlsx(res, path)
 
Save the numeric results of `compute_motif_heatmap` to an Excel file with 4 sheets:
 
- `trials`       : one row per simulation (theta, gamma, trial, seed, t_end,
                   motif_id, class_id, and the off-diagonal weights of the final
                   3x3 E->E matrix, named W_i_j = W[i, j])
- `class_counts` : for each (theta, gamma), number of trials in each of the 8
                   motif classes (class1 ... class8)
- `heatmap_rgb`  : the average RGB color (0-1) at each (theta, gamma) point,
                   i.e. the numbers behind the heatmap
- `legend`       : class id -> label
"""
function export_results_xlsx(res, path::String)
    df = res.df
    ncls = length(MOTIF_LABELS)
    mkpath(dirname(abspath(path)))
 
    # --- Sheet 1: per-trial table -------------------------------------------
    trials = DataFrame(
        theta    = df.theta,
        gamma    = df.gamma,
        trial    = df.trial,
        seed     = df.seed,
        t_end    = df.t_end,
        motif_id = [x === nothing ? missing : x for x in df.motif_id],
        class_id = df.class_id,
    )
    for i in 1:NE, j in 1:NE
        i == j && continue
        trials[!, Symbol("W_$(i)_$(j)")] = [W[i, j] for W in df.final_weights]
    end
    sort!(trials, [:theta, :gamma, :trial])
 
    # --- Sheet 2: class counts per (theta, gamma) ----------------------------
    class_keys = Tuple(Symbol("class$k") for k in 1:ncls)
    rows = NamedTuple[]
    for th in res.unique_thetas, ga in res.unique_gammas
        sel = (df.theta .== th) .& (df.gamma .== ga)
        any(sel) || continue
        cls = df.class_id[sel]
        counts = Tuple(count(c -> !ismissing(c) && c == k, cls) for k in 1:ncls)
        push!(rows, merge((theta = th, gamma = ga, n_trials = count(sel)),
                          NamedTuple{class_keys}(counts)))
    end
    class_counts = DataFrame(rows)
 
    # --- Sheet 3: average RGB heatmap values ---------------------------------
    rgb = DataFrame(theta = Float64[], gamma = Float64[],
                    R = Float64[], G = Float64[], B = Float64[])
    for (i, th) in enumerate(res.unique_thetas), (j, ga) in enumerate(res.unique_gammas)
        push!(rgb, (th, ga, res.heatmap_avg[j, i, 1],
                            res.heatmap_avg[j, i, 2],
                            res.heatmap_avg[j, i, 3]))
    end
 
    # --- Sheet 4: legend ------------------------------------------------------
    legend = DataFrame(class_id = 1:ncls, label = MOTIF_LABELS)
 
    XLSX.writetable(path,
        "trials"       => trials,
        "class_counts" => class_counts,
        "heatmap_rgb"  => rgb,
        "legend"       => legend;
        overwrite = true)
    @info "Saved Excel file: $path"
    return nothing
end

# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================
# rule = 'H'
# df = load_results_table(rule)
# res = compute_motif_heatmap(df)
#
# plt = plot_motif_heatmap(res)
# savefig(plt, plotsdir("motifs_avg_$rule.png"))
# export_results_xlsx(res, datadir("analysis", "motifs_$rule.xlsx"))
# motif_legend()
# ==============================================================================
mkpath(plotsdir())
println("Loading simulation results...")
df = load_results_table(rule)
res = compute_motif_heatmap(df)
plt = plot_motif_heatmap(res)
println("Saving heatmap...")
savefig(plt, plotsdir("motifs_avg_$rule.png"))
export_results_xlsx(res, datadir("analysis", "motifs_$rule.xlsx"))
motif_legend()
println("Analysis complete.")
