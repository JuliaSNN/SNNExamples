# Headless runner for one example script.
#
#   julia --project=. scripts/run_example.jl <script.jl> [--figdir figures] [--max-figs 2]
#
# The script is evaluated top-level expression by top-level expression in Main, with the
# CairoMakie PNG backend. After each expression the current Makie figure is inspected; every
# figure that gets replaced by a new one (and the last one at the end) is saved as
# <figdir>/<script path with / -> __>_<k>.png. The last stdout line is
#   RESULT <pass|fail> <seconds> <number of figures saved> <first error line>
# Exit code 0 on pass, 1 on failure.

ENV["GKSwstype"] = "100"
ENV["JULIA_DEBUG"] = ""

using CairoMakie
CairoMakie.activate!(type = "png", px_per_unit = 1.0)
const Makie = CairoMakie.Makie

const ROOT = normpath(joinpath(@__DIR__, ".."))
const MAX_BYTES = 300_000          # largest PNG written, in bytes

function parse_args(args)
    script = nothing
    figdir = joinpath(ROOT, "figures")
    maxfigs = 2
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--figdir"
            figdir = abspath(args[i+1]); i += 2
        elseif a == "--max-figs"
            maxfigs = parse(Int, args[i+1]); i += 2
        else
            script = abspath(a); i += 1
        end
    end
    script === nothing && error("usage: run_example.jl script.jl [--figdir d] [--max-figs n]")
    return script, figdir, maxfigs
end

mutable struct FigureStore
    dir::String
    stem::String
    max::Int
    last::Any
    saved::Vector{String}
end

function flush_figure!(st::FigureStore, fig)
    fig === nothing && return
    length(st.saved) >= st.max && return
    mkpath(st.dir)
    path = joinpath(st.dir, string(st.stem, "_", length(st.saved) + 1, ".png"))
    try
        # keep every PNG below MAX_BYTES: lower the resolution until the file is small enough
        for ppu in (1.0, 0.8, 0.65, 0.5, 0.4)
            Makie.save(path, fig; px_per_unit = ppu)
            filesize(path) <= MAX_BYTES && break
        end
        push!(st.saved, path)
    catch e
        @warn "could not save figure" exception = (e, catch_backtrace())
    end
end

function track!(st::FigureStore)
    f = try Makie.current_figure() catch; nothing end
    if f !== nothing && f !== st.last
        flush_figure!(st, st.last)
        st.last = f
    end
end

function run_script(script, st)
    src = read(script, String)
    ex = Meta.parseall(src; filename = script)
    task_local_storage(:SOURCE_PATH, script)
    cd(ROOT)
    for e in ex.args
        e isa LineNumberNode && continue
        Base.invokelatest(Core.eval, Main, e)    # new definitions must be visible to the next expression
        Base.invokelatest(track!, st)
    end
    flush_figure!(st, st.last)
end

function first_line(e)
    msg = try sprint(showerror, e) catch; string(typeof(e)) end
    l = split(msg, '\n'; keepempty = false)
    return isempty(l) ? string(typeof(e)) : strip(l[1])
end

function main()
    script, figdir, maxfigs = parse_args(ARGS)
    rel = relpath(script, ROOT)
    stem = replace(splitext(rel)[1], "/" => "__")
    st = FigureStore(figdir, stem, maxfigs, nothing, String[])
    t0 = time()
    status, msg = "pass", ""
    try
        run_script(script, st)
    catch e
        e isa LoadError && (e = e.error)
        status, msg = "fail", first_line(e)
        showerror(stderr, e, catch_backtrace()); println(stderr)
    end
    println(stdout)
    println(stdout, "RESULT\t", status, "\t", round(time() - t0; digits = 1), "\t", length(st.saved), "\t", msg)
    exit(status == "pass" ? 0 : 1)
end

main()
