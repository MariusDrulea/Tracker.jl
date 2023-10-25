using ChainRules: ChainRules, rrule
import Base: sin, cos, sincos

struct Node
    pullback
    parents
end

struct Box <: Number
    value
    node::Node
end

Base.sincos(b::Box) = sin(b), cos(b)
Base.sin(b::Box) = track(sin, b)
Base.cos(b::Box) = track(cos, b)
Base.:+(b1::Box, b2::Box) = track(+, b1, b2)

x = 3.1
bx = Box(x, Node(nothing, ()))
bbx = Box(bx, Node(nothing, ()))

y = 2.4
by = Box(y, Node(nothing, ()))
bby = Box(by, Node(nothing, ()))

function track(f, b::Box)
    println(f)
    v = b.value
    n = b.node
    y, pb = rrule(f, v)
    return Box(y, Node(pb, (n)))
end

function track(f, b1::Box, b2::Box)
    v1, v2 = b1.value, b2.value
    n1, n2 = b1.node, b2.node
    y, pb = rrule(f, v1, v2)
    return Box(y, Node(pb, (n1, n2)))
end

# not a box, call the function on raw value
track(f, x) = f(x)
track(f, x, y) = return f(x, y)

track(sin, x)
sbx = track(sin, bx)
sbbx = track(sin, bbx)

track(+, bx, by)
track(+, bbx, bby)

# back(b::Box, Δ) = back(b.node, Δ)
# 
# function back(n::Node, Δ)
#     Δs = n.pullback(Δ)[2:end]
#     for (p, Δ_) in zip(n.parents, Δs)
#         back(p, Δ_)
#     end
# end

# back((), ::Any) = 