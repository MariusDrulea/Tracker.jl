module Tracker

using MacroTools
using MacroTools: @q, @forward

using ChainRules
using ChainRules: rrule, RuleConfig, HasReverseMode, unthunk
using ForwardDiff
import LogExpFunctions
import NaNMath
import SpecialFunctions

import Printf

import Base: ==
import Base: broadcasted

export TrackedArray, TrackedVector, TrackedMatrix, Params, gradient,
  jacobian, hessian, param, back!, withgradient

include("tracked_types.jl")

# TODO: remove this function after all it's call are removed; see the grad macro
function _forward end

# TODO: this function is used to define gradients for a couple of functions, especially in arrays, which are not used,
# but we might want to define rrules for them, so we keep this code for a while
macro grad(ex)
  @capture(shortdef(ex), (name_(args__) = body_) |
                         (name_(args__) where {T__} = body_)) || error("Need a function definition")
  T == nothing && (T = [])
  isexpr(name, :(::)) || (name = :(::typeof($name)))
  insert!(args, 1+isexpr(args[1], :parameters) , name)
  @q(Tracker._forward($(args...)) where $(T...) = $body) |> esc
end

if !isdefined(Base, :get_extension)
  using Requires
end

@static if !isdefined(Base, :get_extension)
function __init__()
  @require PDMats="90014a1f-27ba-587c-ab20-58faa44d9150" include("../ext/TrackerPDMatsExt.jl")
end
end

include("idset.jl")
include("params.jl")
include("lib/real.jl")
include("lib/array.jl")
include("back.jl")
include("numeric.jl")
include("forward.jl")

# we define this in order to access rrule for broadcasted
struct TrackerRuleConfig <: RuleConfig{HasReverseMode} end
const tracker_rule_cfg = TrackerRuleConfig()
const dummy_broadcast_style = Base.BroadcastStyle(Float64) # requested by rrule for broadcasted, which is only to please Zygote

# dedicated track method for broadcasted
function track(bf::typeof(Base.broadcasted), f::F, xs...; kw...) where F
  @info "Chainrules for $bf($f, ...)"
  y, _back = rrule(tracker_rule_cfg, bf, dummy_broadcast_style, f, data.(xs)...; kw...)
  back = Δ->_back(Δ)[4:end] # TODO: what happens if f is a struct?
  make_tracked(y, back, xs)
end

# Arithmetic operations +, -, *, ^ have a dedicated specializations in ChainRules; are these faster? we use them here
for f in (:+, :-, :*, :/)
  @eval begin
    function track(bf::typeof(Base.broadcasted), ::typeof($f), xs...; kw...)
      @info "Chainrules for $bf($($f), ...), specialized"
      _y, _back = rrule(bf, $f, data.(xs)...; kw...)
      y = Base.materialize(_y)
      back = Δ->_back(Δ)[3:end]
      make_tracked(y, back, xs)
    end
  end
end

# ^2 also has a dedicated specialization in ChainRules
function track(bf::typeof(Base.broadcasted), lp::typeof(Base.literal_pow), ::typeof(^), x::TrackedTypes, ::Val{2})
  @info "Chainrules for $bf($lp, ^, ..., 2), specialized"
  _y, _back = rrule(bf, lp, ^, data(x), Val(2))
  y = Base.materialize(_y)
  back = Δ->_back(Δ)[4:4] # 4:4 because the output shall be a tuple, not a scalar
  make_tracked(y, back, (x,))
end

# TODO: we can better define a method to select the range of interested values, e.g. without NoTangent()
# option1: specialize it for various methods
# option2: simply scan the result and select values without NoTangent(), shall be consecutive

function track(::typeof(Base.getindex), xs...; kw...)
  @assert length(xs) == 2 # the array and the index
  @info "Chainrules for Base.getindex"
  # untracked primal y; also untracked pullback back as we rrule over the data.(xs)
  y, _back = rrule(Base.getindex, data.(xs)...; kw...)
  back = Δ->_back(Δ)[2:2]
  if typeof(xs[1]) <: TrackedTuple # the rrule getindex from Tuples returns a Tangent{..}(result), compared to arrays where it returns directly the result
    back = Δ->(ChainRules.ChainRulesCore.backing(_back(Δ)[2]),)
  end
  make_tracked(y, back, xs[1:1])   
  # TODO: only tracker.(xs[1:1]), tracker(index) is nothing; hm... use the operations on NoTangent() and avoid all this special treatment?
end

function track(f::F, xs...; kw...) where F
  @info "Chainrules for $f"
  # untracked primal y; also untracked pullback back as we rrule over the data.(xs)
  y, _back = rrule(f, data.(xs)...; kw...)
  # TODO: what happens with structs as functions?
  back = Δ->_back(Δ)[2:end]
  make_tracked(y, back, xs)
end


"""
    hook(f, x) -> x′

Hook into gradient backpropagation. `x` is unmodified, but when backpropagating
`f` will be applied to the incoming gradient. For example, `hook(-, x)` will reverse
the sign of the gradient applied to `x`.
"""
hook(f, x) = istracked(x) ? track(hook, f, x) : x
@grad hook(f, x) = data(x), Δ -> (nothing, f(Δ))

"""
    checkpoint(f, args...)

Behaves like `f(args...)`, but avoids storing the intermediate values needed for
calculating gradients. Instead, `f(args...)` will be called again during the
backward pass. This can be used to save memory in larger models.
"""
checkpoint(f, args...) = track(checkpoint, f, args...)

@grad function checkpoint(f, args...)
  data(f(args...)), function (Δ)
    y, back = forward(f, args...)
    (nothing, back(Δ)...)
  end
end

nobacksies(f, x) = track(nobacksies, f, x)
nobacksies(f, xs::Tuple) = map(x -> nobacksies(f, x), xs)
# TODO: do we need to define a rrule for nobacksies?
rrule(::typeof(nobacksies), f::Symbol, x) = data(x), Δ -> error("Nested AD not defined for $f")
rrule(::typeof(nobacksies), f::String, x) = data(x), Δ -> error(f)
# @grad nobacksies(f::Symbol, x) = data(x), Δ -> error("Nested AD not defined for $f")
# @grad nobacksies(f::String, x) = data(x), Δ -> error(f)


# TODO: do we need to define a rrule for identity?
# @grad identity(x) = data(x), Δ -> (Δ,)
rrule(::typeof(identity), x::TrackedTypes) = data(x), Δ->(NoTangent(), Δ)

end #end of module Tracker
