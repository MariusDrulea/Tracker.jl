Pullback = Union{Function, Nothing}

mutable struct _Tracker{T}
  ref::UInt32 # currently used by the backpropagation algo
  pullback::Pullback # a pullback function or nothing for leafs
  parents # cannot declare as Parents type here as this will lead to circular type definitions
  grad::T # TODO: grad appears twice    
  
  _Tracker{T}(pullback, parents) where T = new(0, pullback, parents)
  _Tracker{T}(pullback, parents, grad::T) where T = new(0, pullback, parents, grad)
end

struct TrackedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
    tracker::_Tracker{T}
    grad::A # todo: grad appears twice
    TrackedArray{T,N,A}(data::A, tr::_Tracker{T}) where {T,N,A} = new(data, tr)
    TrackedArray{T,N,A}(data::A, tr::_Tracker{T}, grad::A) where {T,N,A} = new(data, tr, grad)
end

# we can have y = 2*x; x is TrackedArray, and we also want to keep 2 into a type for a nice display of the graph and for debugging purposes.
struct NotTracked{T}
    data::T
end

# Parent = Union{TrackedReal, TrackedArray, TrackedTuple, NotTracked}
Parent = Union{TrackedArray, NotTracked}
Parents = Tuple{Vararg{Parent}}


# outer constructor to call the inner constructor
TrackedArray(x::A, pullback::Pullback, parents::Parents) where {A <: AbstractArray} = 
TrackedArray{eltype(A),ndims(A),A}(x, _Tracker{A}(pullback, parents))

TrackedArray(x::A, pullback::Pullback, parents::Parents, grad::A) where {A <: AbstractArray} = 
TrackedArray{eltype(A),ndims(A),A}(x, _Tracker{A}(pullback, parents, grad))

TrackedArray(x::AbstractArray) = TrackedArray(x, nothing, (), zero(x))

make_tracked(x::AbstractArray, pullback::Pullback, parents::Parents) = TrackedArray(x, pullback, parents)