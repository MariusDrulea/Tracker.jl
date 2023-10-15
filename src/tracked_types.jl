# generic methods; will be specialized for tracked types
tracker(x) = nothing
istracked(x) = tracker(x) ≠ nothing
isleaf(x) = !istracked(x) || isleaf(tracker(x))
grad(x) = grad(tracker(x))
grad(::Nothing) = nothing
data(x) = x

Pullback = Union{Function, Nothing, Missing}

# TODO: update this after refactoring types
"""
  Tracked{T}
  
Structure used to keep the operations applied over variables. 
Represents a node in the graph. To navigate in the graph, use the `f::Call` field. 

# Parameters
  - `ref`: variable used during the graph traversals, how many times we reached a node
  - `f::Call`: the Call object containing the recorded function and arguments; kindly note the pullback function is stored instead
              of the original function; e.g. we store the pullback of + and not the + function itself
  - `isleaf::Bool`: refers to the node in the built graphs; true if the node (tracked object) is leaf
  - `grad::T`: use to store the value of the back-propagated gradient. 
               To further propagate this gradient, let's call it `∇`, the algorithm applies the Jacobian `∇2 = f.func(∇) = J(f_original)*∇` (the pullback). 
               This new gradient is passed to the "children" of `f` stored in `f.args`.               
               Note the gradient is not always stored. 
               For example if the graph is just a straigh-line, no branches, then we simply back-propagate the gradients 
               from the output to the input params. Only the leafs in the graph (our input params) will store gradients in this case.
               See the `function back(x::Tracked, Δ, once)` for more details.
"""
mutable struct _Tracker
  ref::UInt32 # currently used by the backpropagation algo
  pullback::Pullback # a pullback function or nothing for leafs
  parents # cannot declare as Parents type here as this will lead to circular type definitions
  # TODO: grad appears twice    
  # the type of gradient is not known in the forward pass, only in the backward pass
  # this is because we support multiple types in the graph
  grad
  
  _Tracker(pullback, parents) = new(0, pullback, parents)
  _Tracker(pullback, parents, grad) = new(0, pullback, parents, grad)
end

struct Tracked{T} end # TODO: remove this hack; added to ensure a temporary compilation of old code 
struct Call{T} end

struct TrackedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    tracker::_Tracker
    grad # todo: grad appears twice
    TrackedArray{T,N,A}(data::A, tr::_Tracker) where {T,N,A} = new(data, tr)
    TrackedArray{T,N,A}(data::A, tr::_Tracker, grad) where {T,N,A} = new(data, tr, grad)
end

mutable struct TrackedReal{T<:Real} <: Real
  data::T
  tracker::_Tracker
end

struct TrackedTuple{T<:Tuple}
  data::T
  tracker::_Tracker
end

# we can have y = 2*x; x is TrackedArray, and we also want to keep 2 into a type for a nice display of the graph and for debugging purposes.
struct NotTracked{T}
  data::T
end

TrackedTypes = Union{TrackedReal, TrackedArray, TrackedTuple}
# TODO: we have added Any until we properly treat NotTracked
Parent = Union{TrackedReal, TrackedArray, TrackedTuple, NotTracked, Any}
Parents = Tuple{Vararg{Parent}}

istracked(x::_Tracker) = true
isleaf(x::_Tracker) = isnothing(x.pullback)
grad(x::_Tracker) = x.grad

param(x::Number) = TrackedReal(float(x))
param(xs::AbstractArray) = TrackedArray(float.(xs))

param(x::TrackedReal) = track(identity, x)
param(x::TrackedArray) = track(identity, x)

# TODO: where is this code used?
import Adapt: adapt, adapt_structure
adapt_structure(T, xs::TrackedArray) = param(adapt(T, data(xs)))

