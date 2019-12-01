import Base: size, getindex, setindex!, iterate, length, eachindex, CartesianIndices, tail, IndexStyle, copyto!, _sub2ind
using TupleTools

# calculates the length of a SymArray
symarrlength(Nts,Nsyms) = prod(binomial(Nt-1+Nsym, Nsym) for (Nt,Nsym) in zip(Nts,Nsyms))

struct SymArray{Nsyms,T,N,M,VecType<:AbstractVector} <: AbstractArray{T,N}
    data::VecType
    size::NTuple{N,Int}
    Nts::NTuple{M,Int}
    function SymArray{Nsyms,T}(size::Vararg{Int,N}) where {Nsyms,T,N}
        @assert sum(Nsyms)==N
        ii::Int = 0
        f = i -> (ii+=Nsyms[i]; size[ii])
        Nts = ntuple(f,length(Nsyms))
        len = symarrlength(Nts,Nsyms)
        data = Vector{T}(undef,len)
        new{Nsyms,T,N,length(Nsyms),Vector{T}}(data,size,Nts)
    end
    # this creates a SymArray that serves as a view on an existing vector
    function SymArray{Nsyms}(data::VecType,size::Vararg{Int,N}) where {Nsyms,N,VecType<:AbstractVector{T}} where T
        @assert sum(Nsyms)==N
        ii::Int = 0
        f = i -> (ii+=Nsyms[i]; size[ii])
        Nts = ntuple(f,length(Nsyms))
        len = symarrlength(Nts,Nsyms)
        @assert length(data) == len
        new{Nsyms,T,N,length(Nsyms),VecType}(data,size,Nts)
    end
end

size(A::SymArray) = A.size
length(A::SymArray) = length(A.data)

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

@generated _sub2ind(A::SymArray{Nsyms,T,N}, I::Vararg{Int,M}) where {Nsyms,T,N,M} = begin
    N==M || error("_sub2ind(::SymArray): number of indices $M != number of dimensions $N")
    body = quote
        stride::Int = 1
        ind::Int = 1
        Nt::Int = 0
    end
    ii::Int = 0
    for (iN,Nsym) in enumerate(Nsyms)
        Ilocs = ( :( I[$(ii+=1)] ) for _=1:Nsym)
        expr = quote
            Nt = A.Nts[$iN]
            grpind = symgrp_sortedsub2ind(TupleTools.sort(($(Ilocs...),))...)
            ind += (grpind-1) * stride
            # stride is binomial(Nt+Nsym-1,Nsym) (total size of the symmetric block)
            stride *= @symind_binomial(Nt,$Nsym,-1)
        end
        push!(body.args,expr)
    end
    push!(body.args,:( ind ))
    body
end

IndexStyle(::Type{<:SymArray}) = IndexCartesian()
getindex(A::SymArray, i::Int) = A.data[i]
getindex(A::SymArray{Nsyms,T,N}, I::Vararg{Int,N}) where {Nsyms,T,N} = A.data[_sub2ind(A,I...)]
setindex!(A::SymArray, v, i::Int) = A.data[i] = v
setindex!(A::SymArray{Nsyms,T,N}, v, I::Vararg{Int,N}) where {Nsyms,T,N} = (A.data[_sub2ind(A,I...)] = v);

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
