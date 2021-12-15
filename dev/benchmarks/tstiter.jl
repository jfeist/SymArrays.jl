struct TstIter{Nsym}
    size::Int
    "create an iterator that gives i1<=i2<=i3 etc for one index group"
    TstIter(Nsym,size) = new{Nsym}(size)
end
#Base.ndims(::TstIter{Nsym}) where Nsym = Nsym
#Base.eltype(::Type{TstIter{Nsym}}) where Nsym = NTuple{Nsym,Int}
#Base.length(iter::TstIter{Nsym}) where Nsym = symgrp_size(iter.size,Nsym)
#Base.first(iter::TstIter{Nsym}) where Nsym = Nsym > 0 ? ntuple(one,Val(Nsym)) : ntuple(i->i,Val(-Nsym))
Base.first(iter::TstIter{Nsym}) where Nsym = ntuple(one,Val(Nsym)) # : ntuple(i->i,Val(-Nsym))
#Base.last(iter::TstIter{Nsym}) where Nsym = Nsym>0 ? ntuple(i->iter.size,Val(Nsym)) : ntuple(i->iter.size+Nsym+i #= Nsym < 0 here =#,Val(-Nsym))

@inline function Base.iterate(iter::TstIter)
    iterfirst = first(iter)
    iterfirst, iterfirst
end
@inline function Base.iterate(iter::TstIter, state)
    valid, I = __tstinc(iter, state)
    return valid ? (I, I) : nothing
end

@inline __tstinc(iter::TstIter, state::Tuple{Int}) = (state[1] < iter.size, (state[1]+1,))
# function __tstinc(iter::TstIter{Nsym}, state::Tuple{Int,Int}) where Nsym
#     i1, i2 = state
#     if i1 + (Nsym<0) < i2
#         return true, (i1+1, i2)
#     else
#         return i2 < iter.size, (1, i2+1)
#     end
# end
@inline function __tstinc(iter::TstIter{Nsym}, state::NTuple{N,Int}) where {Nsym,N}
    if state[1] + (Nsym<0) < state[2]
        return true, (state[1]+1, Base.tail(state)...)
    end
    valid, I = __tstinc(iter, Base.tail(state))
    # if Nsym<0, lowest possible value for nth index is n
    return valid, (Nsym>0 ? 1 : 1 - Nsym - N, I...)
end


using BenchmarkTools

for n in (10,20,30)
    for d in 2:4
        SI=TstIter(d,n)
        @btime count(x->true,$SI)
    end
    println()
end