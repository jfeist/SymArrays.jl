import Base: size, getindex, setindex!, iterate, length, eachindex, CartesianIndices, tail, IndexStyle
using TupleTools, StaticArrays

function symarrlength(Nts,Nsyms)
    len = 1
    for (Nt,Nsym) in zip(Nts,Nsyms)
        len *= prod(Nt+ii for ii=0:Nsym-1) ÷ factorial(Nsym)
    end
    len
end

struct SymArray{T,N,Nsyms,M} <: AbstractArray{T,N}
    data::Vector{T}
    size::NTuple{N,Int}
    Nts::NTuple{M,Int}
    function SymArray{T,M,Nsyms}(size...) where {T,M,Nsyms}
        @assert sum(Nsyms)==length(size)==M
        ii::Int = 0
        f = i -> (ii+=Nsyms[i]; size[ii])
        Nts = ntuple(f,length(Nsyms))
        len = symarrlength(Nts,Nsyms)
        data = Vector{T}(undef,len)
        new{T,sum(Nsyms),Nsyms,length(Nts)}(data,size,Nts)
    end
end

size(A::SymArray) = A.size
length(A::SymArray) = length(A.data)

SymArray(A::AbstractArray{T,N},Nsyms) where {T,N} = (S = SymArray{T,N,Nsyms}(size(A)...); for (i,I) in enumerate(eachindex(S)); S[i] = A[I]; end; S);

@generated sub2ind(A::SymArray{T,N,Nsyms}, I::Vararg{Int,N}) where {T,N,Nsyms} = begin
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
getindex(A::SymArray{T,N}, i::Int) where {T,N} = A.data[i]
getindex(A::SymArray{T,N}, I::Vararg{Int,N}) where {T,N} = A.data[sub2ind(A,I...)]
setindex!(A::SymArray{T,N}, v, i::Int) where {T,N} = A.data[i] = v
setindex!(A::SymArray{T,N}, v, I::Vararg{Int,N}) where {T,N} = (A.data[sub2ind(A,I...)] = v);

struct SymArrayIter{N}
    lessnext::NTuple{N,Bool}
    sizes::NTuple{N,Int}
    "create an iterator that gives i1<=i2<=i3 etc for each index group"
    SymArrayIter(A::SymArray{T,N,Nsyms,M}) where {T,N,Nsyms,M} = begin
        lessnext = ones(MVector{N,Bool})
        sizes = A.size
        istart = 1
        for (Nt,Nsym) in zip(A.Nts,Nsyms)
            iend = istart+Nsym
            lessnext[iend-1] = false
            istart = iend
        end
        new{N}(Tuple(lessnext),sizes)
    end
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
