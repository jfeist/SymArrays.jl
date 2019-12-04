import Base: size, getindex, setindex!, iterate, length, eachindex, CartesianIndices, tail, IndexStyle, copyto!
using TupleTools

"size of a single symmetric group with Nsym dimensions and size Nt per dimension"
symgrp_size(Nt,Nsym) = binomial(Nt-1+Nsym, Nsym);

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
    function SymArray{Nsyms,T}(size::Vararg{Int,N}) where {Nsyms,T,N}
        Nts = _getNts(Val(Nsyms),size)
        M = length(Nsyms)
        data = Array{T,M}(undef,symgrp_size.(Nts,Nsyms)...)
        new{Nsyms,T,N,M,typeof(data)}(data,size,Nts)
    end
    # this creates a SymArray that serves as a view on an existing vector
    function SymArray{Nsyms}(data::datType,size::Vararg{Int,N}) where {Nsyms,N,datType<:AbstractArray{T}} where T
        Nts = _getNts(Val(Nsyms),size)
        @assert Base.size(data) == symgrp_size.(Nts,Nsyms)
        new{Nsyms,T,N,length(Nsyms),datType}(data,size,Nts)
    end
end

size(A::SymArray) = A.size
length(A::SymArray) = length(A.data)

symgrp_size(S::SymArray{Nsyms}) where Nsyms = symgrp_size.(S.Nts,Nsyms)
symgrp_size(S::SymArray{Nsyms},d::Integer) where Nsyms = symgrp_size(S.Nts[d],Nsyms[d])
symgrps(S::SymArray{Nsyms}) where Nsyms = Nsyms
nsymgrps(S::SymArray{Nsyms,T,N,M}) where {Nsyms,T,N,M} = M

copyto!(S::SymArray,A::AbstractArray) = begin
    @assert size(S) == size(A)
    @inbounds for (i,I) in enumerate(eachindex(S))
        S[i] = A[I]
    end
    S
end

SymArray{Nsyms}(A::AbstractArray{T}) where {Nsyms,T} = (S = SymArray{Nsyms,T}(size(A)...); copyto!(S,A))
# to avoid ambiguity with Vararg "view" constructor above
SymArray{(1,)}(A::AbstractVector{T}) where {T} = (S = SymArray{(1,),T}(size(A)...); copyto!(S,A))

"`SymArr_ifsym(A,Nsyms)` make a SymArray if there is some symmetry (i.e., any of the Nsyms are not 1)"
SymArr_ifsym(A,Nsyms) = all(Nsyms.==1) ? A : SymArray{(Nsyms...,)}(A)

"""based on Base.binomial, but without negative values for n and without overflow checks
(index calculations here should not overflow if the array does not have more elements than an Int64 can represent)"""
function binomial_simple(n::T, k::T) where T<:Integer
    (k < 0 || k > n) && return zero(T)
    (k == 0 || k == n) && return one(T)
    if k > (n>>1)
        k = n - k
    end
    k == 1 && return n
    x::T = nn = n - k + 1
    nn += 1
    rr = 2
    while rr <= k
        x = div(x*nn, rr)
        rr += 1
        nn += 1
    end
    x
end

"""calculate binomial(ii+n+offset,n), equal to prod((ii+j+offset)/j, j=1:n)
This shows up in size and index calculations for arrays with symmetric indices."""
macro symind_binomial(ii,n::Integer,offset::Integer)
    binom = :( $(esc(ii)) + $(offset+1) ) # j=1
    # careful about operation order:
    # first multiply, the product is then always divisible by j
    for j=2:n
        binom = :( ($binom*($(esc(ii))+$(offset+j))) ÷ $j )
    end
    # (N n) = (N N-n) -> (ii+offset+n n) = (ii+offset+n ii+offset)
    # when we do this replacement, we cannot unroll the loop explicitly,
    # but it still turns out to be faster for large n and small (ii+offset)
    binom_func = :( binomial_simple($(esc(ii))+$(n+offset),$(esc(ii))+$offset) )
    # for small n, just return the unrolled calculation directly
    if n < 10
        binom
    else
        # in principle, ii+offset < n, but heuristically use n/2 to ensure that it wins against explicit unrolling
        :( $(esc(ii)) < $(n÷2 - offset) ? $binom_func : $binom )
    end
end

@inline @generated function symgrp_sortedsub2ind(I::Vararg{T,Nsym})::T where {Nsym,T<:Integer}
    indexpr = :( I[1] )
    for ndim = 2:Nsym
        # calculate binomial(i_n+ndim-2,ndim)
        indexpr = :( $indexpr + @symind_binomial(I[$ndim],$ndim,-2) )
    end
    return indexpr
end

@generated _sub2grp(A::SymArray{Nsyms,T,N}, I::Vararg{Int,N}) where {Nsyms,T,N} = begin
    result = ()
    ii::Int = 0
    for (iN,Nsym) in enumerate(Nsyms)
        Ilocs = ( :( I[$(ii+=1)] ) for _=1:Nsym)
        result = (result..., :( symgrp_sortedsub2ind(TupleTools.sort(($(Ilocs...),))...) ) )
    end
    code = :( ($(result...),) )
    code
end

IndexStyle(::Type{<:SymArray}) = IndexCartesian()
getindex(A::SymArray, i::Int) = A.data[i]
getindex(A::SymArray{Nsyms,T,N}, I::Vararg{Int,N}) where {Nsyms,T,N} = A.data[_sub2grp(A,I...)...]
setindex!(A::SymArray, v, i::Int) = A.data[i] = v
setindex!(A::SymArray{Nsyms,T,N}, v, I::Vararg{Int,N}) where {Nsyms,T,N} = (A.data[_sub2grp(A,I...)...] = v);

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
eachindex(A::SymArray) = CartesianIndices(A);
CartesianIndices(A::SymArray) = SymArrayIter(A);

first(iter::SymArrayIter) = CartesianIndex(map(one, iter.sizes))

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

@generated first(iter::SymIndexIter{Nsym}) where Nsym = :( $(ntuple(one,Nsym)) )

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
