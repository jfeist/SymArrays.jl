var documenterSearchIndex = {"docs":
[{"location":"#SymArrays.jl-1","page":"Home","title":"SymArrays.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Modules = [SymArrays]","category":"page"},{"location":"#SymArrays.SymArr_ifsym-Tuple{Any,Any}","page":"Home","title":"SymArrays.SymArr_ifsym","text":"SymArr_ifsym(A,Nsyms) make a SymArray if there is some symmetry (i.e., any of the Nsyms are not 1)\n\n\n\n\n\n","category":"method"},{"location":"#SymArrays.binomial_simple-Union{Tuple{T}, Tuple{T,T}} where T<:Integer","page":"Home","title":"SymArrays.binomial_simple","text":"based on Base.binomial, but without negative values for n and without overflow checks (index calculations here should not overflow if the array does not have more elements than an Int64 can represent)\n\n\n\n\n\n","category":"method"},{"location":"#SymArrays.check_contraction_compatibility-Union{Tuple{nS}, Tuple{nA}, Tuple{NA}, Tuple{Nsymsres}, Tuple{NsymsS}, Tuple{TU}, Tuple{U}, Tuple{T}, Tuple{SymArray{Nsymsres,TU,N,M,VecType} where VecType<:(AbstractArray{T,1} where T) where M where N,Array{T,NA},SymArray{NsymsS,U,N,M,VecType} where VecType<:(AbstractArray{T,1} where T) where M where N,Val{nA},Val{nS}}} where nS where nA where NA where Nsymsres where NsymsS where TU where U where T","page":"Home","title":"SymArrays.check_contraction_compatibility","text":"Check if the arguments correspond to a valid contraction. Do all \"static\" checks at compile time.\n\n\n\n\n\n","category":"method"},{"location":"#SymArrays.symgrp_info-Union{Tuple{T}, Tuple{T,Any}} where T<:SymArray","page":"Home","title":"SymArrays.symgrp_info","text":"return the symmetry group index, the number of symmetric indices in the group, and the first index of the group\n\n\n\n\n\n","category":"method"},{"location":"#SymArrays.@symind_binomial-Tuple{Any,Integer,Integer}","page":"Home","title":"SymArrays.@symind_binomial","text":"calculate binomial(ii+n+offset,n), equal to prod((ii+j+offset)/j, j=1:n) This shows up in size and index calculations for arrays with symmetric indices.\n\n\n\n\n\n","category":"macro"}]
}