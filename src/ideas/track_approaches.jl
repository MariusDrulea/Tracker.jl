# the track currently used in the Tracker
function track(f::F, xs...; kw...) where F
    @info "Chainrules for $f"  
    y, _back = rrule(f, data.(xs)...; kw...)   # untracked primal y; also untracked pullback back as we rrule over the data.(xs)
    back = Δ->_back(Δ)[2:end]
    make_tracked(y, back, xs)
end

# candidate track for higher order derivatives; can lead to infinite recursivity
function candidate_track(f::F, xs...; kw...) where F
    @info "Chainrules for $f, candidate_track"
    y = f(data.(xs)...) # untracked primal
    _, _back = rrule(f, xs...; kw...) # pullback over TrackedReal, TrackedArray etc.
    back = Δ->_back(Δ)[2:end]
    make_tracked(y, back, xs)
end

# the super_track would to the purpose if we @thunk the primal computation in rrule
function super_track(f::F, xs...; kw...) where F
    @info "Chainrules for $f, super_track"
    y = f(data.(xs)...) # untracked primal
    function _back(Δ)
        _, pb = rrule(f, xs...; kw...) # pullback over TrackedReal, TrackedArray etc.
        return pb(Δ)
    end
    back = Δ->_back(Δ)[2:end]
    make_tracked(y, back, xs)
end


# TRACK of track shall do what?
function track(F=track, f, xs...; kw...)
    @info "Chainrules for $f"  
    y, _back = rrule(track, f, xs...; kw...)   # untracked primal y; also untracked pullback back as we rrule over the data.(xs)
    back = Δ->_back(Δ)[2:end]
    make_tracked(y, back, xs)
end

function rrule(track, f, xs...)
    # makes one more node in the graph, y; back on what to do on that node in backward step
    y, back = track(f, xs...) # which track are we calling here?
    function track_pullback(Δ)
        # todo what here?
    end
    return y, track_pullback
end
  
  