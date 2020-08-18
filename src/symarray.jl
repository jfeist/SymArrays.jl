import Base: size, length, ndims, eltype, first, last, ==
import Base: getindex, setindex!, iterate, eachindex, IndexStyle, CartesianIndices, tail, copyto!, fill!
using TupleTools
using Adapt

"size of a single symmetric group with Nsym dimensions and size Nt per dimension"
symgrp_size(Nt,Nsym) = binomial_simple(Nt-1+Nsym, Nsym)
symgrp_size(Nt,::Val{Nsym}) where Nsym = binomial_unrolled(Nt+(Nsym-1),Val(Nsym))

# calculates the length of a SymArray
symarrlength(Nts,Nsyms) = prod(symgrp_size.(Nts,Nsyms))

@generated function _getNts(::Val{Nsyms},size::NTuple{N,Int}) where {Nsyms,N}
    @assert sum(Nsyms)==N
    symdims = cumsum(collect((1,Nsyms...)))
    Nts = [ :( size[$ii] ) for ii in symdims[1:end-1]]
    code = quote
        Nts = ($(Nts...),)
    end
    err = :( error("SymArray: sizes $size not compatible with symmetry numbers $Nsyms") )
    for ii = 1:length(Nsyms)
        dd = [ :( size[$jj] ) for jj=symdims[ii]:symdims[ii+1]-1 ]
        push!(code.args,:( all(($(dd...),) .== Nts[$ii]) || $err ))
    end
    push!(code.args, :( Nts ))
    code
end

struct SymArray{Nsyms,T,N,M,datType<:AbstractArray} <: AbstractArray{T,N}
    data::datType
    size::NTuple{N,Int}
    Nts::NTuple{M,Int}
    function SymArray{Nsyms,T}(::Type{arrType},size::Vararg{Int,N}) where {Nsyms,T,N,arrType}
        Nts = _getNts(Val(Nsyms),size)
        M = length(Nsyms)
        data = arrType{T,M}(undef,symgrp_size.(Nts,Nsyms)...)
        new{Nsyms,T,N,M,typeof(data)}(data,size,Nts)
    end
    function SymArray{Nsyms,T}(size::Vararg{Int,N}) where {Nsyms,T,N}
        SymArray{Nsyms,T}(Array,size...)
    end
    # this creates a SymArray that serves as a view on an existing array
    function SymArray{Nsyms}(data::datType,size::Vararg{Int,N}) where {Nsyms,N,datType<:AbstractArray{T}} where T
        Nts = _getNts(Val(Nsyms),size)
        @assert Base.size(data) == symgrp_size.(Nts,Val.(Nsyms))
        new{Nsyms,T,N,length(Nsyms),datType}(data,size,Nts)
    end
end

size(A::SymArray) = A.size
length(A::SymArray) = length(A.data)

# this is necessary for CUDAnative kernels, but also generally useful
# e.g., adapt(CuArray,S) will return a copy of the SymArray with storage in a CuArray
Adapt.adapt_structure(to, x::SymArray{Nsyms}) where Nsyms = SymArray{Nsyms}(adapt(to,x.data),x.size...)

symgrp_size(S::SymArray{Nsyms}) where Nsyms = symgrp_size.(S.Nts,Nsyms)
symgrp_size(S::SymArray{Nsyms},d::Integer) where Nsyms = symgrp_size(S.Nts[d],Nsyms[d])
symgrps(S) = symgrps(typeof(S))
symgrps(::Type{<:SymArray{Nsyms}}) where Nsyms = Nsyms
nsymgrps(S) = nsymgrps(typeof(S))
nsymgrps(::Type{<:SymArray{Nsyms,T,N,M}}) where {Nsyms,T,N,M} = M

"""storage_type(A): return the underlying storage type of array wrapper types"""
storage_type(A) = storage_type(typeof(A))
storage_type(::Type{T}) where T = T
storage_type(::Type{<:SymArray{Nsyms,T,N,M,datType}}) where {Nsyms,T,N,M,datType} = datType

copyto!(S::SymArray,A::AbstractArray) = begin
    Ainds, Sinds = LinearIndices(A), LinearIndices(S)
    isempty(Ainds) || (checkbounds(Bool, Sinds, first(Ainds)) && checkbounds(Bool, Sinds, last(Ainds))) || throw(BoundsError(S, Ainds))
    @inbounds for (i,I) in zip(Ainds,eachindex(S))
        S[i] = A[I]
    end
    S
end
copyto!(A::AbstractArray,S::SymArray) = begin
    Ainds, Sinds = LinearIndices(A), LinearIndices(S)
    isempty(Sinds) || (checkbounds(Bool, Ainds, first(Sinds)) && checkbounds(Bool, Ainds, last(Sinds))) || throw(BoundsError(A, Sinds))
    @inbounds for (i,I) in zip(Ainds,CartesianIndices(A))
        A[i] = S[I]
    end
    A
end
copyto!(S::SymArray,Ssrc::SymArray) = begin
    @assert symgrps(S) == symgrps(Ssrc)
    @assert size(S) == size(Ssrc)
    copyto!(S.data,Ssrc.data)
    S
end

fill!(S::SymArray,v) = fill!(S.data,v)

==(S1::SymArray,S2::SymArray) = (symgrps(S1),S1.data) == (symgrps(S2),S2.data)

SymArray{Nsyms}(A::AbstractArray{T}) where {Nsyms,T} = (S = SymArray{Nsyms,T}(size(A)...); copyto!(S,A))
# to avoid ambiguity with Vararg "view" constructor above
SymArray{(1,)}(A::AbstractVector{T}) where {T} = (S = SymArray{(1,),T}(size(A)...); copyto!(S,A))

"`SymArr_ifsym(A,Nsyms)` make a SymArray if there is some symmetry (i.e., any of the Nsyms are not 1)"
SymArr_ifsym(A,Nsyms) = all(Nsyms.==1) ? A : SymArray{(Nsyms...,)}(A)

"calculates the contribution of index idim in (i1,...,idim,...,iN) to the corresponding linear index for the group"
symind2ind(i,::Val{dim}) where dim = binomial_unrolled(i+(dim-2),Val(dim))

"calculates the linear index corresponding to the symmetric index group (i1,...,iNsym)"
@inline @generated function symgrp_sortedsub2ind(I::Vararg{T,Nsym})::T where {Nsym,T<:Integer}
    terms2toN = [ :( symind2ind(I[$dim],Val($dim)) ) for dim=2:Nsym ]
    :( +(I[1],$(terms2toN...)) )
end

@generated _sub2grp(A::SymArray{Nsyms,T,N}, I::Vararg{Int,N}) where {Nsyms,T,N} = begin
    result = []
    ii::Int = 0
    for Nsym in Nsyms
        Ilocs = ( :( I[$(ii+=1)] ) for _=1:Nsym)
        push!(result, :( symgrp_sortedsub2ind(TupleTools.sort(($(Ilocs...),))...) ) )
    end
    code = :( ($(result...),) )
    code
end

IndexStyle(::Type{<:SymArray}) = IndexCartesian()
getindex(A::SymArray, i::Int) = A.data[i]
getindex(A::SymArray{Nsyms,T,N}, I::Vararg{Int,N}) where {Nsyms,T,N} = (@boundscheck checkbounds(A,I...); @inbounds A.data[_sub2grp(A,I...)...])
setindex!(A::SymArray, v, i::Int) = A.data[i] = v
setindex!(A::SymArray{Nsyms,T,N}, v, I::Vararg{Int,N}) where {Nsyms,T,N} = (@boundscheck checkbounds(A,I...); @inbounds A.data[_sub2grp(A,I...)...] = v);

eachindex(S::SymArray) = CartesianIndices(S)
CartesianIndices(S::SymArray) = SymArrayIter(S)

@generated function lessnexts(::SymArray{Nsyms}) where Nsyms
    lessnext = ones(Bool,sum(Nsyms))
    istart = 1
    for Nsym in Nsyms
        istart += Nsym
        lessnext[istart-1] = false
    end
    Tuple(lessnext)
end

struct SymArrayIter{N}
    lessnext::NTuple{N,Bool}
    sizes::NTuple{N,Int}
    "create an iterator that gives i1<=i2<=i3 etc for each index group"
    SymArrayIter(A::SymArray{Nsyms,T,N,M}) where {Nsyms,T,N,M} = new{N}(lessnexts(A),A.size)
end
ndims(::SymArrayIter{N}) where N = N
eltype(::Type{SymArrayIter{N}}) where N = NTuple{N,Int}
first(iter::SymArrayIter) = CartesianIndex(map(one, iter.sizes))
last(iter::SymArrayIter) = CartesianIndex(iter.sizes...)

@inline function iterate(iter::SymArrayIter)
    iterfirst = first(iter)
    iterfirst, iterfirst
end
@inline function iterate(iter::SymArrayIter, state)
    valid, I = __inc(state.I, iter.sizes, iter.lessnext)
    valid || return nothing
    return CartesianIndex(I...), CartesianIndex(I...)
end
# increment post check to avoid integer overflow
@inline __inc(::Tuple{}, ::Tuple{}, ::Tuple{}) = false, ()
@inline function __inc(state::Tuple{Int}, size::Tuple{Int}, lessnext::Tuple{Int})
    valid = state[1] < size[1]
    return valid, (state[1]+1,)
end

@inline function __inc(state, sizes, lessnext)
    smax = lessnext[1] ? state[2] : sizes[1]
    if state[1] < smax
        return true, (state[1]+1, tail(state)...)
    end
    valid, I = __inc(tail(state), tail(sizes), tail(lessnext))
    return valid, (1, I...)
end

struct SymIndexIter{Nsym}
    size::Int
    "create an iterator that gives i1<=i2<=i3 etc for one index group"
    SymIndexIter(Nsym,size) = new{Nsym}(size)
end
ndims(::SymIndexIter{Nsym}) where Nsym = Nsym
eltype(::Type{SymIndexIter{Nsym}}) where Nsym = NTuple{Nsym,Int}
length(iter::SymIndexIter{Nsym}) where Nsym = symgrp_size(iter.size,Nsym)
first(iter::SymIndexIter{Nsym}) where Nsym = ntuple(one,Val(Nsym))
last(iter::SymIndexIter{Nsym}) where Nsym = ntuple(i->iter.size,Val(Nsym))

@inline function iterate(iter::SymIndexIter)
    iterfirst = first(iter)
    iterfirst, iterfirst
end
@inline function iterate(iter::SymIndexIter, state)
    valid, I = __inc(state, iter.size)
    valid || return nothing
    return I, I
end
# increment post check to avoid integer overflow
@inline __inc(::Tuple{}, ::Tuple{}) = false, ()
@inline function __inc(state::Tuple{Int}, size::Int)
    valid = state[1] < size
    return valid, (state[1]+1,)
end
@inline function __inc(state::NTuple{N,Int}, size::Int) where N
    if state[1] < state[2]
        return true, (state[1]+1, tail(state)...)
    end
    valid, I = __inc(tail(state), size)
    return valid, (1, I...)
end

function _find_symind(ind::T, ::Val{dim}, high::T) where {dim,T<:Integer}
    dim==1 ? ind+one(T) : searchlast_func(ind, x->symind2ind(x,Val(dim)),one(T),high)
end

"""convert a linear index for a symmetric index group into a group of subindices"""
@generated function ind2sub_symgrp(SI::SymIndexIter{N},ind::T) where {N,T<:Integer}
    code = quote
        ind -= 1
    end
    kis = Symbol.(:k,1:N)
    for dim=N:-1:2
        push!(code.args,:( $(kis[dim]) = _find_symind(ind,Val($dim),T(SI.size)) ))
        push!(code.args,:( ind -= symind2ind($(kis[dim]),Val($dim)) ))
    end
    push!(code.args, :( k1 = ind + 1 ))
    push!(code.args, :( return ($(kis...),)) )
    #display(code)
    code
end

@generated function _grp2sub(A::SymArray{Nsyms,T,N,M}, I::Vararg{Int,M}) where {Nsyms,T,N,M}
    exs = [:( ind2sub_symgrp(SymIndexIter($Nsym,A.Nts[$ii]),I[$ii]) ) for (ii,Nsym) in enumerate(Nsyms)]
    code = :( TupleTools.flatten($(exs...)) )
    #display(code)
    code
end

ind2sub(A::SymArray,ii) = _grp2sub(A,Tuple(CartesianIndices(A.data)[ii])...)
