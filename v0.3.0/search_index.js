var documenterSearchIndex = {"docs":
[{"location":"#SymArrays.jl","page":"Home","title":"SymArrays.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [SymArrays]","category":"page"},{"location":"#SymArrays.SymArr_ifsym-Tuple{Any,Any}","page":"Home","title":"SymArrays.SymArr_ifsym","text":"SymArr_ifsym(A,Nsyms) make a SymArray if there is some symmetry (i.e., any of the Nsyms are not 1)\n\n\n\n\n\n","category":"method"},{"location":"#SymArrays.binomial_simple-Union{Tuple{T}, Tuple{T,T}} where T<:Integer","page":"Home","title":"SymArrays.binomial_simple","text":"based on Base.binomial, but without negative values for n and without overflow checks (index calculations here should not overflow if the array does not have more elements than an Int64 can represent)\n\n\n\n\n\n","category":"method"},{"location":"#SymArrays.check_contraction_compatibility-Union{Tuple{nS}, Tuple{nA}, Tuple{NA}, Tuple{Nsymsres}, Tuple{NsymsS}, Tuple{TU}, Tuple{U}, Tuple{T}, Tuple{SymArray{Nsymsres,TU,N,M,datType} where datType<:AbstractArray where M where N,StridedArray{T, NA},SymArray{NsymsS,U,N,M,datType} where datType<:AbstractArray where M where N,Val{nA},Val{nS}}} where nS where nA where NA where Nsymsres where NsymsS where TU where U where T","page":"Home","title":"SymArrays.check_contraction_compatibility","text":"Check if the arguments correspond to a valid contraction. Do all \"static\" checks at compile time.\n\n\n\n\n\n","category":"method"},{"location":"#SymArrays.contract_symindex!-Union{Tuple{Nsym}, Tuple{sizeS13unit}, Tuple{sizeA13unit}, Tuple{TU}, Tuple{U}, Tuple{T}, Tuple{Array{TU,5},Array{T,3},Val{sizeA13unit},Array{U,3},Val{sizeS13unit},Val{Nsym}}} where Nsym where sizeS13unit where sizeA13unit where TU where U where T","page":"Home","title":"SymArrays.contract_symindex!","text":"A[iAprev,icntrct,iApost] S[iSprev,Icntrct,ISpost] res[iAprev,iApost,iSprev,Icntrct-1,ISpost]\n\n\n\n\n\n","category":"method"},{"location":"#SymArrays.ind2sub_symgrp-Union{Tuple{T}, Tuple{N}, Tuple{SymArrays.SymIndexIter{N},T}} where T<:Integer where N","page":"Home","title":"SymArrays.ind2sub_symgrp","text":"convert a linear index for a symmetric index group into a group of subindices\n\n\n\n\n\n","category":"method"},{"location":"#SymArrays.storage_type-Tuple{Any}","page":"Home","title":"SymArrays.storage_type","text":"storage_type(A)\n\nReturn the type of the underlying storage array for array wrappers.\n\n\n\n\n\n","category":"method"},{"location":"#SymArrays.symgrp_size-Tuple{Any,Any}","page":"Home","title":"SymArrays.symgrp_size","text":"size of a single symmetric group with Nsym dimensions and size Nt per dimension\n\n\n\n\n\n","category":"method"},{"location":"#SymArrays.symgrp_sortedsub2ind-Union{Tuple{Vararg{T,Nsym}}, Tuple{T}, Tuple{Nsym}} where T<:Integer where Nsym","page":"Home","title":"SymArrays.symgrp_sortedsub2ind","text":"calculates the linear index corresponding to the symmetric index group (i1,...,iNsym)\n\n\n\n\n\n","category":"method"},{"location":"#SymArrays.symind2ind-Union{Tuple{dim}, Tuple{Any,Val{dim}}} where dim","page":"Home","title":"SymArrays.symind2ind","text":"calculates the contribution of index idim in (i1,...,idim,...,iN) to the corresponding linear index for the group\n\n\n\n\n\n","category":"method"},{"location":"#SymArrays.which_symgrp-Union{Tuple{T}, Tuple{T,Any}} where T<:SymArray","page":"Home","title":"SymArrays.which_symgrp","text":"return the symmetry group index and the number of symmetric indices in the group\n\n\n\n\n\n","category":"method"}]
}
