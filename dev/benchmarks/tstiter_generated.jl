using BenchmarkTools
using Base: tail

println("Julia version $VERSION")

symgrp_size(Nt,Nsym) = Nsym > 0 ? binomial(Nt-1+Nsym, Nsym) : binomial(Nt, -Nsym)

struct SymIndexIter{Nsym}
    size::Int
end
Base.IteratorSize(::Type{<:SymIndexIter}) = Base.HasLength()
Base.IteratorEltype(::Type{<:SymIndexIter}) = Base.HasEltype()
Base.ndims(::SymIndexIter{Nsym}) where Nsym = Nsym
Base.eltype(::Type{SymIndexIter{Nsym}}) where Nsym = NTuple{Nsym,Int}
Base.length(iter::SymIndexIter{Nsym}) where Nsym = symgrp_size(iter.size,Nsym)
Base.first(iter::SymIndexIter{Nsym}) where Nsym = Nsym > 0 ? ntuple(one,Val(Nsym)) : ntuple(i->i,Val(-Nsym))
Base.last(iter::SymIndexIter{Nsym}) where Nsym = Nsym>0 ? ntuple(i->iter.size,Val(Nsym)) : ntuple(i->iter.size+Nsym+i #= Nsym < 0 here =#,Val(-Nsym))

@inline function Base.iterate(iter::SymIndexIter)
    I = first(iter)
    I, I
end

double_tuple(xs...) = (xs,xs)

@generated function Base.iterate(iter::SymIndexIter{Nsym}, state) where {Nsym}
    newstate = Any[:(state[$i]) for i=1:abs(Nsym)]
    code = Expr(:block, Expr(:meta, :inline))
    for d = 1:abs(Nsym)
        maxv = d==abs(Nsym) ? :(iter.size) : (Nsym>0 ? :(state[$(d+1)]) : :(state[$(d+1)]-1))
        newstate[d] = :( state[$d] + 1 )
        push!(code.args, :( state[$d] < $maxv && return double_tuple($(newstate...)) ))
        newstate[d] = Nsym>0 ? 1 : d
    end
    push!(code.args, :( return nothing ))
    code
end

sumfirsts(iter) = (s=0; for i in iter s+=first(i) end; s)

n = 50
for N in 1:4, M in (N,-N)
    iter = SymIndexIter{M}(n)
    # check that number of elements is correct
    @assert count(x->true,iter) == length(iter)

    print(M>0 ? "M =  $M: " : "M = $M: ")
    @btime sumfirsts($iter)
end
