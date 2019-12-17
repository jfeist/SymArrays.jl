# Functions for contracting two arrays, both for general StridedArrays as well as
# for SymArrays. we only have a few specialized functions that we have needed up
# to now, but carefully optimize each of them

using TensorOperations
using LinearAlgebra

# Array[i]*SymArray[(i,j,k)]
# indices 1, 2, and 3 are exchangeable here
function contract(A::StridedVector{T},S::SymArray{(3,),U},n::Union{Val{1},Val{2},Val{3}}) where {T,U}
    TU = promote_type(T,U)
    @assert size(S,1) == length(A)
    res = SymArray{(2,),TU}(size(S,1),size(S,2))
    contract!(res,A,S,n)
end

# We know that $j\leq k$ (because $R$ is itself exchange symmetric)
# \begin{align}
# R_{jk} &= \sum_{i=1}^N g_i S_{ijk}
# \end{align}

# Matrix elements represented by $S_{ijk}$:
# \begin{equation}
# \begin{cases}
# S_{ijk}, S_{ikj}, S_{jik}, S_{jki}, S_{kij}, S_{kji} & i<j<k\\
# S_{ijk}, S_{jik}, S_{jki} & i<j=k\\
# S_{ijk}, S_{ikj}, S_{kij} & i=j<k\\
# S_{ijk} & i=j=k
# \end{cases}
# \end{equation}
# We only need to take the contributions that show up for each $R_{jk}$

# Array[i]*SymArray[(i,j,k)]
# indices 1, 2, and 3 are exchangeable here
function contract!(res::SymArray{(2,),TU}, A::StridedVector{T}, S::SymArray{(3,),U}, n::Union{Val{1},Val{2},Val{3}}) where {T,U,TU}
    # only loop over S once, and put all the values where they should go
    # R[j,k] = sum_i A[i] B[i,j,k]
    # S[i,j,k] with i<=j<=k represents the 6 (not always distinct) terms: Bijk, Bikj, Bjik, Bjki, Bkij, Bkji
    # since R[j,k] is also exchange symmetric, we only need to calculate j<=k
    # this means we only have to check each adjacent pair of indices and include
    # their permutation if not equal, but keeping the result indices ordered
    @assert size(S,1) == length(A)
    @assert size(S,1) == size(res,1)
    res.data .= 0
    @inbounds for (v,inds) in zip(S.data,CartesianIndices(S))
        i,j,k = Tuple(inds)
        res[j,k] += v*A[i]
        # have to include i<->j
        if i<j res[i,k] += v*A[j] end
        # have to include i<->k, but keep indices of res sorted
        if j<k res[i,j] += v*A[k] end
    end
    res
end

# Array[k]*SymArray[(i,j),k]
function contract(A::StridedVector{T},S::SymArray{(2,1),U},n::Val{3}) where {T,U}
    TU = promote_type(T,U)
    sumsize = length(A)
    @assert sumsize == size(S,3)
    res = SymArray{(2,),TU}(size(S,1),size(S,2))
    contract!(res,A,S,n)
end

# Array[k]*SymArray[(i,j),k]
function contract!(res::SymArray{(2,),TU},A::StridedVector{T},S::SymArray{(2,1),U},::Val{3}) where {T,U,TU}
    # use that S[(i,j),k] == S[I,k] (i.e., the two symmetric indices act like a "big" index)
    mul!(res.data,reshape(S.data,:,length(A)),A)
    res
end

# Array[i]*SymArray[(i,j),k)]
# since indices 1 and 2 are exchangeable here, use this
function contract(A::StridedVector{T},S::SymArray{(2,1),U},n::Union{Val{1},Val{2}}) where {T,U}
    TU = promote_type(T,U)
    @assert size(S,1) == length(A)
    # the result is a normal 2D array
    res = Array{TU,2}(undef,size(S,2),size(S,3))
    contract!(res,A,S,n)
end

# Array[i]*SymArray[(i,j),k]
# since indices 1 and 2 are exchangeable here, use this
function contract!(res::StridedArray{TU,2},A::StridedVector{T},S::SymArray{(2,1),U},::Union{Val{1},Val{2}}) where {T,U,TU}
    # only loop over S once, and put all the values where they should go
    @assert size(A,1) == size(S,1)
    @assert size(res,1) == size(S,1)
    @assert size(res,2) == size(S,3)
    res .= zero(TU)
    @inbounds for (v,inds) in zip(S.data,CartesianIndices(S))
        i1,i2,i3 = Tuple(inds)
        res[i2,i3] += v*A[i1]
        # if i1 != i2, we have to add the equal contribution from S[i2,i1,i3]
        if i1 != i2
            res[i1,i3] += v*A[i2]
        end
    end
    res
end

# Array[i]*SymArray[(i,j)]
# this is symmetric in i1 and i2
function contract(A::StridedVector{T},S::SymArray{(2,),U},n::Union{Val{1},Val{2}}) where {T,U}
    TU = promote_type(T,U)
    @assert size(S,1) == length(A)
    # the result is a normal 1D vector
    res = Vector{TU}(undef,size(S,2))
    contract!(res,A,S,n)
end

# Array[i]*SymArray[(i,j)]
# this is symmetric in i1 and i2
function contract!(res::StridedVector{TU},A::StridedVector{T},S::SymArray{(2,),U},::Union{Val{1},Val{2}}) where {T,U,TU}
    @assert size(A,1) == size(S,1)
    @assert size(res,1) == size(S,1)
    res .= zero(TU)
    # only loop over S once, and put all the values where they should go
    for (v,inds) in zip(S.data,CartesianIndices(S))
        i1,i2 = Tuple(inds)
        res[i2] += v*A[i1]
        # if i1 != i2, we have to add the contribution from S[i2,i1]
        if i1 != i2
            res[i1] += v*A[i2]
        end
    end
    res
end

# Array[i_n]*Array[i1,i2,i3,...,iN]
function contract(A::StridedVector{T},B::StridedArray{U,N},::Val{n}) where {T,U,N,n}
    TU = promote_type(T,U)
    @assert 1 <= n <= N

    resdims = size(B)[1:N .!= n]
    res = similar(B,TU,resdims)

    A = convert(AbstractArray{TU},A)
    B = convert(AbstractArray{TU},B)
    contract!(res,A,B,Val{n}())
end

mygemv!(args...) = BLAS.gemv!(args...)

function _contract_middle!(res,A,B)
    @inbounds for k=1:size(B,3)
        mul!(@view(res[:,k]), @view(B[:,:,k]), A)
    end
end

# Array[i_n]*Array[i1,i2,i3,...,iN]
function contract!(res::StridedArray{TU},A::StridedVector{TU},B::StridedArray{TU,N},::Val{n}) where {TU,N,n}
    nsum = length(A)
    @assert size(B,n) == nsum
    @assert ndims(res)+1 == ndims(B)
    ii = 0
    for jj = 1:ndims(B)
        jj==n && continue
        ii += 1
        @assert size(B,jj) == size(res,ii)
    end

    if n==1      # A[i]*B[i,...]
        mygemv!('T',one(TU),reshape(B,nsum,:),A,zero(TU),vec(res))
    elseif n==N  # B[...,i]*A[i]
        mygemv!('N',one(TU),reshape(B,:,nsum),A,zero(TU),vec(res))
    else
        rightsize = prod(size(B,i) for i=n+1:N)
        Br = reshape(B,:,nsum,rightsize)
        resr = reshape(res,:,rightsize)
        _contract_middle!(resr,A,Br)
    end
    res
end

# Array[i_n]*Array[i1,i2,i3,...,iN]
function contract!(res::StridedArray{TU,Nres},A::StridedArray{TU,NA},B::StridedArray{TU,NB},::Val{nA},::Val{nB}) where {TU,Nres,NA,NB,nA,nB}
    # use specialized code for 1D-arrays if possible
    if NA==1
        @assert nA==1
        return contract!(res,A,B,Val(nB))
    elseif NB==1
        @assert nB==1
        return contract!(res,B,A,Val(nA))
    end

    nsum = size(A,nA)
    @assert size(B,nB) == nsum
    @assert Nres == NA + NB - 2
    ii = 0
    for jj = 1:NA
        jj==nA && continue
        ii += 1
        @assert size(A,jj) == size(res,ii)
    end
    for jj = 1:NB
        jj==nB && continue
        ii += 1
        @assert size(B,jj) == size(res,ii)
    end

    if nA==NA && nB==1      # A[...,i]*B[i,...]
        resAsize = prod(ntuple(i->size(A,i),Val(NA-1)))
        mul!(reshape(res,resAsize,:),reshape(A,resAsize,nsum),reshape(B,nsum,:))
    else
        packedsize(M,::Val{n}) where n = begin
            nM1 = prod(ntuple(i->size(M,i),Val(n-1)))
            nM2 = size(M,n)
            nM3 = prod(ntuple(i->size(M,n+i),Val(ndims(M)-n)))
            (nM1, nM2, nM3)
        end
        # just use @tensor after reshaping to pack together
        NAps = packedsize(A,Val(nA))
        NBps = packedsize(B,Val(nB))
        Nresps = (NAps[1],NAps[3],NBps[1],NBps[3])
        Ap = reshape(A,NAps...)
        Bp = reshape(B,NBps...)
        resp = reshape(res,Nresps...)
        @tensor resp[iA1,iA3,iB1,iB3] = Ap[iA1,ii,iA3] * Bp[iB1,ii,iB3]
    end
    res
end

"""return the symmetry group index and the number of symmetric indices in the group"""
@inline which_symgrp(S::T,nS) where T<:SymArray = which_symgrp(T,nS)
@inline @generated function which_symgrp(::Type{<:SymArray{Nsyms}},nS) where Nsyms
    grps = ()
    for (ii,Nsym) in enumerate(Nsyms)
        grps = (grps...,ntuple(_->ii,Nsym)...)
    end

    quote
        ng = $grps[nS]
        ng, $Nsyms[ng]
    end
end

"""Check if the arguments correspond to a valid contraction. Do all "static" checks at compile time."""
@generated function check_contraction_compatibility(res::SymArray{Nsymsres,TU}, A::StridedArray{T,NA}, S::SymArray{NsymsS,U}, ::Val{nA}, ::Val{nS}) where {T,U,TU,NsymsS,Nsymsres,NA,nA,nS}
    promote_type(T,U) <: TU || error("element types not compatible: T = $T, U = $U, TU = $TU")

    contracted_group, Nsym_ctrgrp = which_symgrp(S,nS)
    NsymsA_contracted = ntuple(_->1,NA-1)
    if Nsym_ctrgrp == 1
        NsymsS_contracted = TupleTools.deleteat(NsymsS,contracted_group)
    else
        NsymsS_contracted = Base.setindex(NsymsS,Nsym_ctrgrp-1,contracted_group)
    end
    Nsymsres_check = (NsymsA_contracted...,NsymsS_contracted...)
    # assure that symmetry structure is compatible
    errmsg = "symmetry structure not compatible:"
    errmsg *= "\nNA = $NA, NsymsS = $NsymsS, nA = $nA, nS = $nS"
    errmsg *= "\nNsymsres = $Nsymsres, expected Nsymssres = $Nsymsres_check"
    Nsymsres == Nsymsres_check || error(errmsg)

    Sresinds = ((1:nS-1)...,(nS+1:sum(NsymsS))...)
    Aresinds = ((1:nA-1)...,(nA+1:NA)...)
    sizerescheck = ((i->:(sizeA[$i])).(Aresinds)...,
                    (i->:(sizeS[$i])).(Sresinds)...)
    code = quote
        sizeA = size(A)
        sizeS = size(S)
        @assert sizeA[$nA] == sizeS[$nS]
        @assert size(res) == ($(sizerescheck...),)
    end
    #display(code)
    code
end

"""
A[iAprev,icntrct,iApost]
S[iSprev,Icntrct,ISpost]
res[iAprev,iApost,iSprev,Icntrct-1,ISpost]
"""
@generated function contract_symindex!(res::Array{TU,5}, A::Array{T,3}, ::Val{sizeA13unit}, S::Array{U,3}, ::Val{sizeS13unit}, ::Val{Nsym}) where {T,U,TU,sizeA13unit,sizeS13unit,Nsym}
    # Nsym-dimensional index tuple
    iS2s = Tuple(Symbol.(:iS2_,1:Nsym))
    # combined (Nsym-1)-dimensional index for the Nsym possible permutations
    iSm2s = Tuple(Symbol.(:iSm2_,1:Nsym))

    iAsetters = Expr(:block)
    iAusers = Expr(:block)
    for n = 1:Nsym
        chk = n==1 ? true : :( $(iS2s[n-1])<$(iS2s[n]) )
        syminds = TupleTools.deleteat(iS2s,n)
        push!(iAsetters.args, :( $chk && ($(iSm2s[n]) = symgrp_sortedsub2ind($(syminds...))) ))
        push!(iAusers.args, :( $chk && (res[iA1,iA3,iS1,$(iSm2s[n]),iS3] += v * A[iA1,$(iS2s[n]),iA3]) ))
    end

    iA1max = sizeA13unit[1] ? 1 : :(size(A,1))
    iA3max = sizeA13unit[2] ? 1 : :(size(A,3))
    iS1max = sizeS13unit[1] ? 1 : :(size(S,1))
    iS3max = sizeS13unit[2] ? 1 : :(size(S,3))

    code = quote
        # size of iterated index is middle index of A
        iterdimS = SymIndexIter($Nsym,size(A,2))

        res .= zero(TU)
        @inbounds for iS3 = 1:$iS3max
            for (iS2,IS) = enumerate(iterdimS)
                ($(iS2s...),) = Tuple(IS)
                $iAsetters
                for iS1 = 1:$iS1max
                    v = S[iS1,iS2,iS3]
                    for iA3 = 1:$iA3max
                        for iA1 = 1:$iA1max
                            $iAusers
                        end
                    end
                end
            end
        end
    end
    code
end

@generated function contract!(res::SymArray{Nsymsres,TU}, A::StridedArray{T,NA}, S::SymArray{NsymsS,U}, ::Val{nA}, ::Val{nS}) where {T,U,TU,NsymsS,Nsymsres,NA,nA,nS}
    contracted_group, Nsym_ctrgrp = which_symgrp(S,nS)
    sizeA13unit = (nA==1,nA==NA)
    sizeS13unit = (contracted_group==1,contracted_group==length(NsymsS))
    newsize_centered_expr(sizeA::Symbol,nA::Int,NA::Int) = begin
        sizeAs = [:( $sizeA[$ii] ) for ii=1:NA]
        t1 = nA > 1 ? :( *($(sizeAs[1:nA-1]...)) ) : 1
        t3 = nA < NA ? :( *($(sizeAs[nA+1:NA]...)) ) : 1
        :( ($t1, $sizeA[$nA], $t3) )
    end

    code = quote
        # first check that all the sizes are compatible etc
        check_contraction_compatibility(res,A,S,Val($nA),Val($nS))

        sizeA = size(A)
        sizeAp = $(newsize_centered_expr(:sizeA,nA,NA))
        Apacked = reshape(A,sizeAp)

        grpsizeS = symgrp_size.(S.Nts,Val.(NsymsS))
        sizeSp = $(newsize_centered_expr(:grpsizeS, contracted_group, length(NsymsS)))
        Spacked = reshape(S.data,sizeSp)

        if $Nsym_ctrgrp > 1
            size_respS1 = symgrp_size(S.Nts[$contracted_group],Val($(Nsym_ctrgrp-1)))
            respacked = reshape(res.data,sizeAp[1],sizeAp[3],sizeSp[1],size_respS1,sizeSp[3])
            contract_symindex!(respacked,Apacked,Val($sizeA13unit),Spacked,Val($sizeS13unit),Val($Nsym_ctrgrp))
        else
            # iS2 is a single (not symmetric) dimension and it disappears after summing
            # we have res[iS1,iS3,iA1,iA3] = S_iS1,k,iS3 * A_iA1,k,iA3
            respacked = reshape(res.data,sizeAp[1],sizeAp[3],sizeSp[1],sizeSp[3])
            @tensor respacked[iA1,iA3,iS1,iS3] = Apacked[iA1,ii,iA3] * Spacked[iS1,ii,iS3]
        end
    end
    #display(code)
    code
end
