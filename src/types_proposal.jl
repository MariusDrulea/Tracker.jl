# Tr is the type of Tracker_
struct TrackedArray{T,N,A<:AbstractArray{T,N}, Tr} <: AbstractArray{T,N}
    data::A
    tracker::Tr
    grad::A # todo: grad appears twice
    TrackedArray{T,N,A,Tr}(data::A, tr::Tr) where {T,N,A,Tr} = new(data, tr)
    TrackedArray{T,N,A,Tr}(data::A, tr::Tr, grad::A) where {T,N,A,Tr} = new(data, tr, grad)
end

# we can have y = 2*x; x is TrackedArray, and we also want to keep 2 into a type for a nice display of the graph and for debugging purposes.
struct NotTracked{T}
    data::T
end

Pullback = Union{Function, Nothing}
# Parent = Union{TrackedReal, TrackedArray, TrackedTuple, NotTracked}
Parent = Union{TrackedArray, NotTracked}
Parents = Tuple{Vararg{Parent}}

mutable struct _Tracker{T, Pb<:Pullback, Prs<:Parents}
    ref::UInt32 # currently used by the backpropagation algo
    pullback::Pb # a pullback function or nothing for leafs
    parents::Prs
    grad::T # TODO: grad appears twice    
    
    _Tracker{T,Pb,Prs}(pullback::Pb, parents::Prs) where {T,Pb,Prs} = new(0, pullback, parents)
    _Tracker{T,Pb,Prs}(pullback::Pb, parents::Prs, grad::T) where {T,Pb,Prs} = new(0, pullback, parents, grad)
end


# outer constructor to call the inner constructor
TrackedArray(x::A, pullback::Pb, parents::Prs) where {A <: AbstractArray, Pb <: Pullback, Prs <: Parents} = 
TrackedArray{eltype(A),ndims(A),A, _Tracker{A, Pb, Parents}}(x, _Tracker{A, Pb, Parents}(pullback, parents))

TrackedArray(x::A, pullback::Pb, parents::Prs, grad::A) where {A <: AbstractArray, Pb <: Pullback, Prs <: Parents} = 
TrackedArray{eltype(A),ndims(A),A, _Tracker{A, Pb, Parents}}(x, _Tracker{A, Pb, Parents}(pullback, parents, grad))

TrackedArray(x::AbstractArray) = TrackedArray(x, nothing, (), zero(x))

make_tracked(x::AbstractArray, pullback, parents) = TrackedArray(x, pullback, parents)