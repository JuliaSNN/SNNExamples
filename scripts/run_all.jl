# Run examples, one process each, with a timeout, at most `--jobs` in parallel.
#
#   julia --project=. scripts/run_all.jl [--jobs 6] [--timeout 300] [--figdir figures] [--max-figs 2] [paths...]
#
# `paths` are scripts or folders (default: everything listed in scripts/examples.toml that is not
# marked `not run`). Output of each script goes to _logs/<name>.log, the outcome to
# _logs/results.tsv (path, status, seconds, figures, first error line). Status is one of
# pass, fail, timeout. `julia scripts/make_status.jl` turns results.tsv into STATUS.md.

using TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
const LOGS = joinpath(ROOT, "_logs")

function parse_args(args)
    jobs, tmo, figdir, maxfigs, paths = 6, 300, joinpath(ROOT, "figures"), 2, String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--jobs"; jobs = parse(Int, args[i+1]); i += 2
        elseif a == "--timeout"; tmo = parse(Int, args[i+1]); i += 2
        elseif a == "--figdir"; figdir = abspath(args[i+1]); i += 2
        elseif a == "--max-figs"; maxfigs = parse(Int, args[i+1]); i += 2
        else push!(paths, a); i += 1 end
    end
    return jobs, tmo, figdir, maxfigs, paths
end

function expand(paths)
    out = String[]
    for p in paths
        ap = isabspath(p) ? p : joinpath(pwd(), p)
        if isdir(ap)
            for (d, _, fs) in walkdir(ap), f in fs
                endswith(f, ".jl") && push!(out, relpath(joinpath(d, f), ROOT))
            end
        else
            push!(out, relpath(ap, ROOT))
        end
    end
    return sort(unique(out))
end

manifest() = TOML.parsefile(joinpath(@__DIR__, "examples.toml"))["example"]

function default_targets()
    return sort([e["path"] for e in manifest() if get(e, "status", "run") == "run"])
end

# per-script timeout: `--timeout` unless the manifest asks for a longer one
function script_timeout(rel, tmo)
    for e in manifest()
        e["path"] == rel && return max(tmo, get(e, "timeout", 0))
    end
    return tmo
end

function run_one(rel, tmo, figdir, maxfigs)
    mkpath(LOGS)
    log = joinpath(LOGS, replace(splitext(rel)[1], "/" => "__") * ".log")
    runner = joinpath(@__DIR__, "run_example.jl")
    cmd = `nice -n 10 timeout -k 10 $(tmo) julia --project=$(ROOT) -t 2 $runner $(joinpath(ROOT, rel)) --figdir $figdir --max-figs $maxfigs`
    t0 = time()
    p = run(pipeline(ignorestatus(cmd); stdout = log, stderr = log, stdin = devnull))
    dt = round(time() - t0; digits = 1)
    code = p.exitcode
    status, nfig, msg = "fail", "0", ""
    lines = readlines(log)
    idx = findlast(l -> startswith(l, "RESULT\t"), lines)
    if code == 124 || code == 137
        status, msg = "timeout", "no result after $(tmo) s"
    elseif idx !== nothing
        f = split(lines[idx], '\t'; limit = 5)
        status, dt, nfig, msg = f[2], parse(Float64, f[3]), f[4], length(f) >= 5 ? f[5] : ""
    elseif dt >= tmo - 2
        status, msg = "timeout", "no result after $(tmo) s"
    else
        status, msg = "fail", "no RESULT line (exit code $(code)); see log"
    end
    return (rel, status, dt, nfig, replace(msg, '\t' => ' '))
end

function main()
    jobs, tmo, figdir, maxfigs, paths = parse_args(ARGS)
    targets = isempty(paths) ? default_targets() : expand(paths)
    mkpath(LOGS)
    res = String[]
    results = Dict{String,Any}()
    lock_ = ReentrantLock()
    resfile = joinpath(LOGS, "results.tsv")
    old = Dict{String,String}()
    isfile(resfile) && for l in readlines(resfile); old[split(l, '\t')[1]] = l; end
    sem = Base.Semaphore(jobs)
    @sync for rel in targets
        @async begin
            Base.acquire(sem)
            try
                r = run_one(rel, script_timeout(rel, tmo), figdir, maxfigs)
                lock(lock_) do
                    old[r[1]] = join(string.(r), '\t')
                    println(rpad(r[2], 8), lpad(string(r[3]), 8), "  ", r[1], "  ", r[5])
                    flush(stdout)
                    open(resfile, "w") do io
                        for k in sort(collect(keys(old))); println(io, old[k]); end
                    end
                end
            finally
                Base.release(sem)
            end
        end
    end
end

main()
