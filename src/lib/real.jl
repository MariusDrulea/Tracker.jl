# TODO: for chainrules
import ChainRules: rrule
import Base: +, -, *, /

# outer constructor to call the inner constructor
TrackedReal(x::Real) = TrackedReal(x, _Tracker(nothing, (), zero(x)))

data(x::TrackedReal) = x.data
tracker(x::TrackedReal) = x.tracker

make_tracked(x::Real, pb::Pullback, pa::Parents) = TrackedReal(x, _Tracker(pb, pa, zero(x)))

function back!(x::TrackedReal; once = true)
    isinf(x) && error("Loss is Inf")
    isnan(x) && error("Loss is NaN")
    return back!(x, 1, once = once)
end

function update!(x::TrackedReal, Δ)
  x.data += data(Δ)
  tracker(x).grad = 0
  return x
end

function Base.show(io::IO, x::TrackedReal)
  T = get(io, :typeinfo, Any)
  show(io, data(x))
  T <: TrackedReal || print(io, " (tracked)")
end

Base.decompose(x::TrackedReal) = Base.decompose(data(x))

Base.copy(x::TrackedReal) = x

Base.convert(::Type{TrackedReal{T}}, x::TrackedReal{T}) where T = x

Base.convert(::Type{TrackedReal{T}}, x::Real) where T = TrackedReal(convert(T, x))

Base.convert(::Type{TrackedReal{T}}, x::TrackedReal{S}) where {T,S} =
  error("Not implemented: convert tracked $S to tracked $T")

(T::Type{<:TrackedReal})(x::Real) = convert(T, x)

for op in [:(==), :≈, :<, :(<=)]
  @eval Base.$op(x::TrackedReal, y::Real) = Base.$op(data(x), y)
  @eval Base.$op(x::Real, y::TrackedReal) = Base.$op(x, data(y))
  @eval Base.$op(x::TrackedReal, y::TrackedReal) = Base.$op(data(x), data(y))
end

Base.eps(x::TrackedReal) = eps(data(x))
Base.eps(::Type{TrackedReal{T}}) where T = eps(T)

for f in :[isinf, isnan, isfinite].args
  @eval Base.$f(x::TrackedReal) = Base.$f(data(x))
end

Printf.fix_dec(x::TrackedReal, n::Int, a...) = Printf.fix_dec(data(x), n, a...)
Printf.tofloat(x::TrackedReal) = Printf.tofloat(data(x))

Base.float(x::TrackedReal) = x

Base.promote_rule(::Type{TrackedReal{S}},::Type{T}) where {S,T} =
  TrackedReal{promote_type(S,T)}

using Random

for f in :[rand, randn, randexp].args
  @eval Random.$f(rng::AbstractRNG,::Type{TrackedReal{T}}) where {T} = param(rand(rng,T))
end

# TODO: for chainrules
for op in [:+, :-, :*, :/]
  @eval Base.$op(x::TrackedReal, y::TrackedReal) = track(Base.$op, x, y)
end

for op in [:sin, :cos]
  @eval Base.$op(x::TrackedReal) = track(Base.$op, x)
end

# Eliminating ambiguity
import Base:^

# TODO: take care of this POW
^(a::TrackedReal, b::Integer) = track(^, a, b)

# Tuples

# outer constructor to call the inner constructor
TrackedTuple(x::T) where T<:Tuple = TrackedTuple{T}(x, _Tracker(nothing, (), zero(x)))

data(xs::TrackedTuple) = xs.data
tracker(xs::TrackedTuple) = xs.tracker

accum!(x::Tuple, Δ::Tuple) = accum!.(x, Δ)
init_grad(x::Tuple) = init_grad.(x)
zero_grad!(x::Tuple) = zero_grad!.(x)

make_tracked(xs::Tuple, pb::Pullback, pa::Parents) = TrackedTuple(xs, _Tracker(pb, pa, zero.(xs)))

function Base.show(io::IO, xs::TrackedTuple)
  show(io, data(xs))
  print(io, " (tracked)")
end

Base.length(x::TrackedTuple) = length(data(x))

Base.getindex(xs::TrackedTuple, i::Integer) = track(getindex, xs, i)

# Array collection

function collect(xs)
  xs = Base.collect(xs)
  track(Call(collect, (tracker.(xs),)), data.(xs))
end

function scan(c::Call{typeof(collect)})
  foreach(scan, c.args[1])
end

function back_(c::Call{typeof(collect)}, Δ, once)
  foreach((x, d) -> back(x, d, once), c.args[1], data(Δ))
end

function back_(g::Grads, c::Call{typeof(collect)}, Δ)
  foreach((x, Δ) -> back(g, x, Δ), c.args[1], Δ)
end

collectmemaybe(xs::AbstractArray{>:TrackedReal}) = collect(xs)
collectmemaybe(xs::AbstractArray{<:TrackedReal}) = collect(xs)

SpecialFunctions.logabsgamma(x::TrackedReal) = track(SpecialFunctions.logabsgamma, x)