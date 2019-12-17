using CuArrays
using CuArrays: cudims, @cuindex
using CUDAnative

mygemv!(tA,alpha,A::CuArray,args...) = CuArrays.CUBLAS.gemv!(tA,alpha,A,args...)
_contract_middle!(res::CuArray,A,B) = (@tensor res[i,k] = B[i,j,k] * A[j])

import Base: collect
# for SymArrays on the GPU, collect should convert to a SymArray on the CPU
# to convert to a "normal" Array, you then have to apply collect again
collect(S::SymArray{Nsyms,T,N,M,datType}) where {Nsyms,T,N,M,datType<:CuArray} = SymArray{Nsyms}(collect(S.data),S.size...)

@generated function cuda_contraction_kernel(res, A, S, SI::SymIndexIter{Nsymres}) where {Nsymres}
    # calculate res[iA1,iA3,iS1,iSm2,iS3] = ∑_iA2 A[iA1,iA2,iA3] * S[iS1,iS2,iS3]
    # where iSm2 = (i1,i2,...,iNsymres) and iS2 = sorted(iA2,i1,i2,i3...,iNsymres)
    code = quote
        I = @cuindex(res)
        iA1,iA3,iS1,iSm2,iS3 = I
        ISm2 = ind2sub_symgrp(SI, iSm2)
        res[I...] = zero(eltype(res))
    end
    for n = 0:Nsymres
        iAstart = n==0 ? 1 : :(ISm2[$n]+1)
        iAend = n<Nsymres ? :(ISm2[$(n+1)]) : :(size(A,2))
        iprev = [:( ISm2[$i] ) for i=1:n]
        ipost = [:( ISm2[$i] ) for i=n+1:Nsymres]
        cc = :( for iA2 = $iAstart:$iAend
                    iS2 = symgrp_sortedsub2ind($(iprev...),iA2,$(ipost...))
                    res[I...] += A[iA1,iA2,iA3]*S[iS1,iS2,iS3]
                end)
        push!(code.args,cc)
    end
    push!(code.args,:(return))
    #display(code)
    :( @inbounds $code )
end

function contract_symindex!(res::CuArray{TU,5}, A::CuArray{T,3}, ::Val{sizeA13unit}, S::CuArray{U,3}, ::Val{sizeS13unit}, ::Val{Nsym}) where {T,U,TU,sizeA13unit,sizeS13unit,Nsym}
    blk, thr = cudims(res)
    SI = SymIndexIter(Nsym-1,size(A,2))
    @cuda blocks=blk threads=thr cuda_contraction_kernel(res,A,S,SI)
end