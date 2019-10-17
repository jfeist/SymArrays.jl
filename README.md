# SymArrays.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jfeist.github.io/SymArrays.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jfeist.github.io/SymArrays.jl/dev)
[![Build Status](https://travis-ci.com/jfeist/SymArrays.jl.svg?branch=master)](https://travis-ci.com/jfeist/SymArrays.jl)
[![Codecov](https://codecov.io/gh/jfeist/SymArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jfeist/SymArrays.jl)

This package provides some tools to efficiently store arrays with exchange symmetries, i.e., arrays where exchanging two indices leaves the value unchanged. It stores the underlying data in a flat vector and provides mappings that allow to address it as a "normal" `AbstractArray{T,N}`. To generate a new one with undefined data, use
```julia
S = SymArray{Nsyms,T}(dims...)
```
where `NSyms` is a tuple that indicates the size of each group of exchangeable indices (which have to be adjacent for simplicity), `T` is the element type (e.g., `Float64` or `ComplexF64`), and `dims` are the dimensions of the array (which have to fulfill `length(dims)==sum(Nsyms)`. As an example
```julia
S = SymArray{(3,1,2,1),Float64}(10,10,10,3,50,50,50)
```
declares an array `S[(i,j,k),l,(m,n),o]` where any permutation of `(i,j,k)` leaves the value unchanged, as does any permutation of `(m,n)`. Note that interchangeable indices obviously have to have the same size (this is currently not checked explicitly in the input!).

## TODO:
- Check for consistency of array size with symmetry groups on construction
- Check bounds violations when accessing matrix elements 
- Allow specification and treatment of Hermitian indices, where any permutation conjugates the result (possibly only for 2 indices at a time?).
