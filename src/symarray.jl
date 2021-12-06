import Base: size, length, ndims, eltype, first, last, ==, parent
import Base: getindex, setindex!, iterate, eachindex, IndexStyle, CartesianIndices, tail, copyto!, fill!
using LinearAlgebra
using TupleTools
using Adapt

"size of a single symmetric group with Nsym dimensions and size Nt per dimension. Nsym<0 means antisymmetric"
symgrp_size(Nt,Nsym) = Nsym > 0 ? binomial_simple(Nt-1+Nsym, Nsym) : binomial_simple(Nt, -Nsym)
symgrp_size(Nt,::Val{Nsym}) where Nsym = Nsym > 0 ? binomial_unrolled(Nt+(Nsym-1),Val(Nsym)) : binomial_unrolled(Nt,Val(-Nsym))

# calculates the length of a SymArray
symarrlength(Nts,Nsyms) = prod(symgrp_size.(Nts,Nsyms))

@generated function _getNts(::Val{Nsyms},size::NTuple{N,Int}) where {Nsyms,N}
    @assert sum(abs.(Nsyms))==N
    symdims = cumsum(collect((1,abs.(Nsyms)...)))
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

parent(A::SymArray) = A.data
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

"""
    storage_type(A)

Return the type of the underlying storage array for array wrappers.
"""
storage_type(A) = storage_type(typeof(A))
storage_type(T::Type) = (P = parent_type(T); P===T ? T : storage_type(P))

parent_type(T::Type{<:AbstractArray}) = T
parent_type(::Type{<:PermutedDimsArray{T,N,perm,iperm,AA}}) where {T,N,perm,iperm,AA} = AA
parent_type(::Type{<:LinearAlgebra.Transpose{T,S}}) where {T,S} = S
parent_type(::Type{<:LinearAlgebra.Adjoint{T,S}}) where {T,S} = S
parent_type(::Type{<:SubArray{T,N,P}}) where {T,N,P} = P
parent_type(::Type{<:SymArray{Nsyms,T,N,M,datType}}) where {Nsyms,T,N,M,datType} = datType

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

Base.similar(src::SymArray{Nsyms}) where Nsyms = SymArray{Nsyms}(similar(parent(src)),size(src)...)
Base.copy(src::SymArray{Nsyms}) where Nsyms = SymArray{Nsyms}(copy(parent(src)),size(src)...)

fill!(S::SymArray,v) = fill!(S.data,v)

==(S1::SymArray,S2::SymArray) = (symgrps(S1),S1.data) == (symgrps(S2),S2.data)

SymArray{Nsyms}(A::AbstractArray{T}) where {Nsyms,T} = (S = SymArray{Nsyms,T}(size(A)...); copyto!(S,A))
# to avoid ambiguity with Vararg "view" constructor above
SymArray{(1,)}(A::AbstractVector{T}) where {T} = (S = SymArray{(1,),T}(size(A)...); copyto!(S,A))

"`SymArr_ifsym(A,Nsyms)` make a SymArray if there is some symmetry (i.e., any of the Nsyms are not 1)"
SymArr_ifsym(A,Nsyms) = all(Nsyms.==1) ? A : SymArray{(Nsyms...,)}(A)

"calculates the contribution of index idim in (i1,...,idim,...,iN) to the corresponding linear index for the group"
symind2ind(i,::Val{dim}) where dim = binomial_unrolled(i+(dim-2),Val(dim))
asymind2ind(i,::Val{dim}) where dim = binomial_unrolled(i-1,Val(dim))

"calculates the linear index corresponding to the symmetric index group (i1,...,iNsym)"
@inline @generated function symgrp_sortedsub2ind(I::Vararg{T,Nsym})::T where {Nsym,T<:Integer}
    terms2toN = [ :( symind2ind(I[$dim],Val($dim)) ) for dim=2:Nsym ]
    :( +(I[1],$(terms2toN...)) )
end

"calculates the linear index corresponding to the antisymmetric index group (i1,...,iNsym)"
@inline @generated function asymgrp_sortedsub2ind(I::Vararg{T,Nsym})::T where {Nsym,T<:Integer}
    terms2toN = [ :( asymind2ind(I[$dim],Val($dim)) ) for dim=2:Nsym ]
    :( +(I[1],$(terms2toN...)) )
end

function _sub2grp_code(Nsyms)
    permsigns = []
    result = []
    ii::Int = 0
    code = quote end
    for (isym,Nsym) in enumerate(Nsyms)
        Isrtd = Symbol(:Isrtd,isym)
        Ilocs = [ :(I[$(ii+=1)]) for _=1:abs(Nsym) ]
        Iloc = :( ($(Ilocs...),) )
        #println(Iloc)
        if Nsym > 0
            push!(code.args, :( $Isrtd = TupleTools.sort($Iloc) ))
            push!(permsigns, 1)
            push!(result, :( symgrp_sortedsub2ind($Isrtd...) ))
        else
            psym = Symbol(:p,isym)
            push!(code.args, :( ($psym, $Isrtd) = sort_track_parity($Iloc)) )
            push!(permsigns, psym)
            push!(result, :( asymgrp_sortedsub2ind($Isrtd...) ))
        end
    end
    push!(code.args, :( *($(permsigns...)), ($(result...),) ))
    code
end

@generated _sub2grp(A::SymArray{Nsyms,T,N}, I::Vararg{Int,N}) where {Nsyms,T,N} = _sub2grp_code(Nsyms)

IndexStyle(::Type{<:SymArray}) = IndexCartesian()
getindex(A::SymArray, i::Int) = A.data[i]
getindex(A::SymArray{Nsyms,T,N}, I::Vararg{Int,N}) where {Nsyms,T,N} = begin
    @boundscheck checkbounds(A,I...)
    permsign, grpinds = _sub2grp(A,I...)
    permsign == 0 && return zero(T)
    @inbounds permsign * A.data[grpinds...]
end
setindex!(A::SymArray, v, i::Int) = A.data[i] = v
setindex!(A::SymArray{Nsyms,T,N}, v, I::Vararg{Int,N}) where {Nsyms,T,N} = begin
    @boundscheck checkbounds(A,I...)
    permsign, grpinds = _sub2grp(A,I...)
    @boundscheck permsign == 0 && throw(ArgumentError("indices $I do not exist for SymArray{$Nsyms} due to exchange antisymmetry."))
    @inbounds A.data[grpinds...] = permsign * v
end

eachindex(S::SymArray) = CartesianIndices(S)
CartesianIndices(S::SymArray) = SymArrayIter(S)

struct SymArrayIter{Nsyms,N}
    size::NTuple{N,Int}
    SymArrayIter(A::SymArray{Nsyms,T,N}) where {Nsyms,T,N} = new{Nsyms,N}(A.size)
end

Base.IteratorSize(::Type{<:SymArrayIter}) = Base.HasLength()
Base.IteratorEltype(::Type{<:SymArrayIter}) = Base.HasEltype()
Base.ndims(::SymArrayIter{Nsym,N}) where {Nsym,N} = N
Base.eltype(::Type{SymArrayIter{Nsym,N}}) where {Nsym,N} = NTuple{N,Int}
Base.length(iter::SymArrayIter{Nsym,N}) where {Nsym,N} = symarrlength(_getNts(Val(Nsyms),iter.size),Nsyms)
# make these generated functions so the code is simply the constant final tuple
@generated Base.first(::SymArrayIter{Nsyms}) where Nsyms = CartesianIndex(TupleTools.flatten(first.(SymIndexIter.(Nsyms,Nsyms))))
@generated Base.last(::SymArrayIter{Nsyms}) where Nsyms = CartesianIndex(TupleTools.flatten(last.(SymIndexIter.(Nsyms,Nsyms))))

@inline function Base.iterate(iter::SymArrayIter)
    cI = first(iter)
    cI, Tuple(cI)
end

cartind_and_tuple(xs...) = (CartesianIndex(xs),xs)

# do these with a generated function - when there is antisymmetry, this is much
# easier than trying to come up with logic and data structures for efficient
# tuple tail recursion
@generated function Base.iterate(iter::SymArrayIter{Nsyms,N}, state::NTuple{N,Int}) where {Nsyms,N}
    newstate = Any[:(state[$i]) for i=1:N]
    code = Expr(:block, Expr(:meta, :inline))
    # global dimension index
    D = 0
    for Nsym in Nsyms
        for d = 1:abs(Nsym)
            D += 1
            maxv = d==abs(Nsym) ? :(iter.size[$D]) : (Nsym>0 ? :(state[$(D+1)]) : :(state[$(D+1)]-1))
            newstate[D] = :( state[$D] + 1 )
            push!(code.args, :( state[$D] < $maxv && return cartind_and_tuple($(newstate...)) ))
            newstate[D] = Nsym>0 ? 1 : d
        end
    end
    push!(code.args, :(return nothing))
    code
end

struct SymIndexIter{Nsym}
    size::Int
    "create an iterator that gives i1<=i2<=i3 etc for one index group"
    SymIndexIter(Nsym,size) = new{Nsym}(size)
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

function _find_symind(ind::T, ::Val{dim}, high::T) where {dim,T<:Integer}
    dim==1 ? ind+one(T) : searchlast_func(ind, x->symind2ind(x,Val(dim)),one(T),high)
end
function _find_asymind(ind::T, ::Val{dim}, high::T) where {dim,T<:Integer}
    dim==1 ? ind+one(T) : searchlast_func(ind, x->asymind2ind(x,Val(dim)),T(dim),high)
end

"""convert a linear index for a symmetric index group into a group of subindices"""
@generated function ind2sub_symgrp(SI::SymIndexIter{N},ind::T) where {N,T<:Integer}
    code = quote
        ind -= 1
    end
    kis = Symbol.(:k,1:abs(N))
    for dim=abs(N):-1:2
        if N>0
            push!(code.args,:( $(kis[dim]) = _find_symind(ind,Val($dim),T(SI.size)) ))
            push!(code.args,:( ind -= symind2ind($(kis[dim]),Val($dim)) ))
        else
            push!(code.args,:( $(kis[dim]) = _find_asymind(ind,Val($dim),T(SI.size)) ))
            push!(code.args,:( ind -= asymind2ind($(kis[dim]),Val($dim)) ))
        end
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

@inline ind2sub(A, ii) = Tuple(CartesianIndices(A)[ii])
@inline ind2sub(A::SymArray,ii) = _grp2sub(A,ind2sub(A.data,ii)...)
