struct TrackedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    tracker::_Tracker{A}
    grad::A # todo: grad appears twice
    TrackedArray{T,N,A}(t::_Tracker{A}, data::A) where {T,N,A} = new(t, data)
    TrackedArray{T,N,A}(t::_Tracker{A}, data::A, grad::A) where {T,N,A} = new(t, data, grad)
end

mutable struct _Tracker{T, Pb, Parents}
    pullback::Pb # a pullback function or nothing for leafs
    parents::Parents # Tuple of Union{TrackedArray, TrackedReal, TrackedTuple, NotTracked}
    grad::T # TODO: grad appears twice    
    ref::UInt32 # currently used by the backpropagation algo
    isleaf::Bool #TODO: isnt's pullback == nothing enough for the isleaf check?
    
    _Tracker{T}(pullback) where T = new(0, pullback, false)
    _Tracker{T}(pullback, grad::T) where T = new(0, pullback, false, grad)
    _Tracker{T}(nothing, grad::T) where T = new(0, nothing, true, grad)
end

# we can have y = 2*x; x is TrackedArray, and we also want to keep 2 into a type for a nice display of the graph and for debugging purposes.
struct NotTracked{T}
    data::T
end
