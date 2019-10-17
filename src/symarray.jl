import Base: size, getindex, setindex!, iterate, length, eachindex, CartesianIndices, tail, IndexStyle, copyto!
using TupleTools

function symarrlength(Nts,Nsyms)
    len = 1
    for (Nt,Nsym) in zip(Nts,Nsyms)
        len *= prod(Nt+ii for ii=0:Nsym-1) ÷ factorial(Nsym)
    end
    len
end

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

@generated sub2ind(A::SymArray{Nsyms,T,N}, I::Vararg{Int,N}) where {Nsyms,T,N} = begin
    body = quote
        stride::Int = 1
        ind::Int = 1
        Nt::Int = 0
    end
    ii::Int = 0
    for (iN,Nsym) in enumerate(Nsyms)
        isyms = Symbol.(:i,1:Nsym)
        Ilocs = ( :( I[$(ii+=1)] ) for _=1:Nsym)
        indexpr = :( $(isyms[1])-1 )
        for (inum,isym) in enumerate(isyms[2:end])
            prodterms = (:( $isym + $(j-1) ) for j=0:inum)
            indexpr = :( $indexpr + *($(prodterms...)) ÷ $(factorial(inum+1)) )
        end
        strideterms = ( :( Nt+$ii ) for ii=0:Nsym-1)
        expr = quote
            Nt = A.Nts[$iN]
            ($(isyms...),) = TupleTools.sort(($(Ilocs...),))
            ind += $indexpr * stride
            stride *= *($(strideterms...)) ÷ $(factorial(Nsym))
        end
        push!(body.args,expr)
    end
    push!(body.args,:( ind ))
    body
end

IndexStyle(::Type{<:SymArray}) = IndexCartesian()
getindex(A::SymArray, i::Int) = A.data[i]
getindex(A::SymArray{Nsyms,T,N}, I::Vararg{Int,N}) where {Nsyms,T,N} = A.data[sub2ind(A,I...)]
setindex!(A::SymArray, v, i::Int) = A.data[i] = v
setindex!(A::SymArray{Nsyms,T,N}, v, I::Vararg{Int,N}) where {Nsyms,T,N} = (A.data[sub2ind(A,I...)] = v);

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
