using Tracker: Tracker, param, TrackedArray, TrackedReal, TrackedTypes, track, back!, back, candidate_track, super_track
using ChainRules: ChainRules, rrule, NoTangent, @thunk, unthunk

##
function f(x::Real)
    println("f($x)")
    x*sin(x) + cos(x) + tan(x)
end

function ChainRules.rrule(f, x)
    y = @thunk f(x)
    pb = Δ->(NoTangent(), Δ*(1+x+3x^2+x^3))
    return y, pb
end

##
f(t::TrackedReal) = track(f, t)

t = param(2.0)
ft = f(t)
back!(ft)
t.tracker.grad

## restart REPL to redefine f
# make sure you thunk the primal
# also try the super_track
f(t::TrackedReal) = candidate_track(f, t)

t = param(2.0)
ft = f(t)


