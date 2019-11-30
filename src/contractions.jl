# Functions for contracting two arrays, both for general StridedArrays as well as
# for SymArrays. we only have a few specialized functions that we have needed up
# to now, but carefully optimize each of them

using TensorOperations
using LinearAlgebra
using CUDAapi

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

if has_cuda_gpu()
    using CuArrays
    mygemv!(tA,alpha,A::CuArray,args...) = CuArrays.CUBLAS.gemv!(tA,alpha,A,args...)
    _contract_middle!(res::CuArray,A,B) = (@tensor res[i,k] = B[i,j,k] * A[j])
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

function contract_gen!(res::SymArray{NsymsC,TU}, A::Array{T,NA}, S::SymArray{NsymsS,U}, ::Val{nA}, ::Val{nS}) where {T,U,TU,NsymsS,NsymsC,NA,nA,nS}
    # only loop over S once, and put all the values where they should go

    # we contract one dimension from each array
    @assert sum(NsymsC) == sum(NsymsS)+NA-2
    
    sizeA = size(A)
    sizeS = size(S)
    @assert sizeA[nA] == sizeS[nS]
    NtsS = S.Nts
    contracted_group, Nsym_ctrgrp, nS_1 = symgroup(S,nS)
    NsymsA_contracted = ntuple(_->1,NA-1)
    NtsA_contracted = TupleTools.deleteat(sizeA,nA)
    if Nsym_ctrgrp == 1
        NsymsS_contracted = TupleTools.deleteat(NsymsS,contracted_group)
        NtsS_contracted = TupleTools.deleteat(NtsS,contracted_group)
    else
        NsymsS_contracted = Base.setindex(NsymsS,Nsym_ctrgrp-1,contracted_group)
        NtsS_contracted = NtsS
    end
    NsymsC_check = (NsymsA_contracted...,NsymsS_contracted...)
    NtsC_check = (NtsA_contracted...,NtsS_contracted...)
    @assert NsymsC_check == NsymsC
    @assert NtsC_check == res.Nts
    
    if NA>1
        Aindexp = ntuple(i->i==nA ? 1 : Colon(),NA)
        Aloopinds = CartesianIndices(A[Aindexp...])
    else
        Aloopinds = ((),)
    end 
    
    res.data .= 0
    @inbounds for (v,indsS) in zip(S.data,CartesianIndices(S))
        IS = Tuple(indsS)
        isum = IS[nS_1]
        IresS = TupleTools.deleteat(IS,nS_1)
        for indsresA in Aloopinds
            IresA = Tuple(indsresA)
            Aprevinds = IresA[1:nA-1]
            Apostinds = IresA[nA:end]
            res[IresA...,IresS...] += v * A[Aprevinds...,isum,Apostinds...]
        end
        for nS_exc = nS_1 : nS_1+Nsym_ctrgrp-2
            if IS[nS_exc] < IS[nS_exc+1]
                isum = IS[nS_exc+1]
                IresS = TupleTools.deleteat(IS,nS_exc+1)
                for indsresA in Aloopinds
                    IresA = Tuple(indsresA)
                    Aprevinds = IresA[1:nA-1]
                    Apostinds = IresA[nA:end]
                    res[IresA...,IresS...] += v * A[Aprevinds...,isum,Apostinds...]
                end
            end
        end
    end
    res
end