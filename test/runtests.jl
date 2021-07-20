using Test
using SymArrays
using SymArrays: symarrlength, _sub2grp, which_symgrp, SymIndexIter, ind2sub_symgrp, sort_track_parity
using TensorOperations
using Random
using CUDA
using cuTENSOR
using Combinatorics: levicivita

CUDA.allowscalar(false)

function test_ind2sub(SI::SymIndexIter)
    correct = true
    for (ii, I) in enumerate(SI)
        correct &= I == ind2sub_symgrp(SI,ii)
    end
    correct
end

@testset "SymArrays.jl" begin
    @testset "sort_track_parity" begin
        tst_sortpar = () -> begin
            sort_parity(x) = levicivita(sortperm(x))*allunique(x)
            allok = true
            for ii = 1:1000
                n = rand(1:10)
                # replace = true so we can get parity 0
                a = rand(1:100, n)
                p1, as1 = sort_track_parity(Tuple(a))
                p2, as2 = sort_parity(a), Tuple(sort(a))
                # do not use @test here so we do not get 1000 tests in the count
                allok &= (p1,as1) == (p2,as2)
            end
            allok
        end
        @test tst_sortpar()
    end

    @testset "SymIndexIter" begin
        for d in 1:4
            for n in (10,15,30,57)
                for Nsym = (-d,d)
                    SI = SymIndexIter(Nsym,n)
                    @test count(i->true,SI) == length(SI)
                    @test test_ind2sub(SI)
                end
            end
        end
    end

    @testset "SymArray" begin
        @test symarrlength((3,6,4,3),(3,2,1,3)) == 8400

        S = SymArray{(2,),Float64}(5,5)
        @test nsymgrps(S) == 1
        @test symgrps(S) == (2,)
        @test _sub2grp(S,2,5) == _sub2grp(S,5,2)
        @test _sub2grp(S,3,4) != _sub2grp(S,5,3)
        # _sub2grp has to have same number of arguments as size of N
        @test_throws MethodError _sub2grp(S,3,5,1)
        # indexing the array allows having additional 1s at the end
        @test S[3,5,1] == S[3,5]
        @test_throws BoundsError S[3,5,3]
        # this calculation gives an index for data that is within bounds, but should not be legal
        # make sure this is caught
        @test_throws BoundsError S[0,6]
        @test (S[1,5] = 2.; S[1,5] == 2.)
        @test_throws BoundsError S[0,6] = 2.

        S = SymArray{(3,1,2,2),Float64}(3,3,3,2,4,4,4,4)
        @test nsymgrps(S) == 4
        @test symgrps(S) == (3,1,2,2)
        @test size(S) == (3,3,3,2,4,4,4,4)
        @test length(S) == 2000
        # iterating over all indices should give only the distinct indices,
        # i.e., give the same number of terms as the array length
        @test sum(1 for s in S) == length(S)
        @test sum(1 for I in eachindex(S)) == length(S)

        # calculating the linear index when iterating over Cartesian indices should give sequential access to the array
        @test 1:length(S) == [(LinearIndices(S.data)[_sub2grp(S,Tuple(I)...)[2]...] for I in eachindex(S))...]

        @testset "_sub2grp" begin
            # test that permuting exchangeable indices accesses the same array element
            i1  = _sub2grp(S,1,2,3,1,4,3,2,1)
            @test _sub2grp(S,2,3,1,1,4,3,2,1) == i1
            @test _sub2grp(S,2,3,1,1,3,4,2,1) == i1
            @test _sub2grp(S,2,3,1,1,3,4,1,2) == i1
            @test _sub2grp(S,2,1,3,1,4,3,1,2) == i1
            # make sure that swapping independent indices gives a different array element
            @test _sub2grp(S,2,1,3,1,4,1,3,2) != i1
            @test _sub2grp(S,1,2,3,2,4,3,2,1) != i1
            @test _sub2grp(S,1,2,3,1,1,3,2,4) != i1

            SI = eachindex(S)
            @test first(SI) == CartesianIndex(1,1,1,1,1,1,1,1)

            NN = 4
            maxNdim = 40
            for Ndim = 1:maxNdim
                S = SymArray{(Ndim,),Float64}(ntuple(i->NN,Ndim)...)
                Is = ntuple(i->NN,Ndim)
                @test _sub2grp(S,Is...) == (1, size(S.data))
                Is = ntuple(i->1,Ndim)
                @test _sub2grp(S,Is...) == (1, (1,))
            end
        end

        A = rand(5,5)
        @test A' != A
        @test SymArray{(1,1)}(A) == A
        @test SymArray{(2,)}(A) != A

        A = A + A'
        # the standard equality test uses iteration over the arrays, (a,b) in zip(A,B),
        # which only accesses the actually different indices in SymArrays
        @test SymArray{(2,)}(A) != A
        # broadcasting goes over all indices
        @test all(SymArray{(2,)}(A) .== A)

        # views on existing vector-like types
        x = rand(3)
        S = SymArray{(2,)}(x,2,2)
        @test S.data === x
        fill!(S,8.)
        @test all(S .== 8.)

        S2 = SymArray{(2,)}(0*x,2,2)
        copyto!(S2,S)
        @test S == S2

        x = 5:10
        S = SymArray{(2,)}(x,3,3)
        @test S.data === x

        A = rand(10)
        S = SymArray{(1,)}(A,10)
        @test S.data === A

        # but if called without sizes, the copy constructor should be used
        S = SymArray{(1,)}(A)
        @test S.data !== A
    end

    @testset "which_symgrp" begin
        S = SymArray{(3,2),Float64}(5,5,5,3,3)
        @test which_symgrp(S,1) == (1,3)
        @test which_symgrp(S,3) == (1,3)
        @test which_symgrp(S,4) == (2,2)
        @test which_symgrp(S,5) == (2,2)
    end

    @testset "Contractions" begin
        @testset "Manual" begin
            N,M,O = 3, 5, 8
            for T in (Float64,ComplexF64)
                A = rand(T,N)

                for U in (Float64,ComplexF64)
                    for (n,dims) in enumerate([(N,M,O),(O,N,M),(M,O,N)])
                        B = rand(U,dims...)
                        n==1 && (@tensor D1[j,k] := A[i]*B[i,j,k])
                        n==2 && (@tensor D1[j,k] := A[i]*B[j,i,k])
                        n==3 && (@tensor D1[j,k] := A[i]*B[j,k,i])
                        D2 = contract(A,B,Val(n))
                        @test D1 ≈ D2
                        if T==U
                            contract!(D2,A,B,Val(n))
                            @test D1 ≈ D2
                        else
                            @test_throws MethodError contract!(D2,A,B,Val(n))
                        end
                    end

                    S = SymArray{(3,),U}(N,N,N)
                    rand!(S.data)
                    B = collect(S)
                    @test contract(A,S,Val(1)) == contract(A,S,Val(2))
                    @test contract(A,S,Val(1)) == contract(A,S,Val(3))
                    for n = 1:3
                        @test collect(contract(A,S,Val(n))) ≈ contract(A,B,Val(n))
                    end

                    S = SymArray{(2,1),U}(N,N,N)
                    rand!(S.data)
                    B = collect(S)
                    @test contract(A,S,Val(1)) == contract(A,S,Val(2))
                    for n = 1:3
                        @test collect(contract(A,S,Val(n))) ≈ contract(A,B,Val(n))
                    end
                    @test !(collect(contract(A,S,Val(1))) ≈ collect(contract(A,S,Val(3))))

                    S = SymArray{(2,),U}(N,N)
                    rand!(S.data)
                    B = collect(S)
                    @test contract(A,S,Val(1)) ≈ contract(A,S,Val(2))
                    for n = 1:2
                        @test collect(contract(A,S,Val(n))) ≈ contract(A,B,Val(n))
                    end
                end
            end
        end
        @testset "Generated" begin
            # use small dimension sizes here so the tests do not take too long
            N,M,O = 4, 5, 6
            arrTypes = has_cuda_gpu() ? (Array,CuArray) : (Array,)

            for arrType in arrTypes
                for T in (Float64,ComplexF64)
                    A = rand(T,N,M,O) |> arrType
                    for U in (Float64,ComplexF64)
                        S = SymArray{(2,3,1),U}(arrType,M,M,N,N,N,O)
                        @test storage_type(S) <: arrType
                        
                        rand!(S.data)
                        # first collect GPU->CPU, then SymArray -> Array
                        B = collect(collect(S)) |> arrType
                        @test storage_type(B) <: arrType

                        TU = promote_type(U,T)
                        res21 = SymArray{(1,1,1,3,1),TU}(arrType,N,O,M,N,N,N,O)
                        contract!(res21,A,S,Val(2),Val(1))
                        @tensor res21_tst[i,k,l,m,n,o,p] := A[i,j,k] * B[j,l,m,n,o,p]
                        @test collect(collect(res21)) ≈ collect(res21_tst)
                        if T==U
                            contract!(res21_tst,A,B,Val(2),Val(1))
                            @test collect(collect(res21)) ≈ collect(res21_tst)
                        end

                        res13 = SymArray{(1,1,2,2,1),TU}(arrType,M,O,M,M,N,N,O)
                        contract!(res13,A,S,Val(1),Val(3))
                        @tensor res13_tst[j,k,l,m,n,o,p] := A[i,j,k] * B[l,m,i,n,o,p]
                        @test collect(collect(res13)) ≈ collect(res13_tst)
                        if T==U
                            contract!(res13_tst,A,B,Val(1),Val(3))
                            @test collect(collect(res13)) ≈ collect(res13_tst)
                        end

                        # dimension 3, 4, and 5 should be equivalent
                        contract!(res13,A,S,Val(1),Val(4))
                        @test collect(collect(res13)) ≈ collect(res13_tst)
                        contract!(res13,A,S,Val(1),Val(5))
                        @test collect(collect(res13)) ≈ collect(res13_tst)
                    end

                    A = rand(T,N) |> arrType
                    S = SymArray{(2,3,1),T}(arrType,N,N,N,N,N,N)
                    rand!(S.data)
                    # first collect GPU->CPU, then SymArray -> Array
                    B = collect(collect(S)) |> arrType

                    res11 = SymArray{(1,3,1),T}(arrType,N,N,N,N,N)
                    contract!(res11,A,S,Val(1),Val(1))
                    @tensor res11_tst[j,k,l,m,n] := A[i] * B[i,j,k,l,m,n]
                    @test collect(collect(res11)) ≈ collect(res11_tst)
                    contract!(res11_tst,A,B,Val(1),Val(1))
                    @test collect(collect(res11)) ≈ collect(res11_tst)

                    res13 = SymArray{(2,2,1),T}(arrType,N,N,N,N,N)
                    contract!(res13,A,S,Val(1),Val(3))
                    @tensor res13_tst[j,k,l,m,n] := A[i] * B[j,k,i,l,m,n]
                    @test collect(collect(res13)) ≈ collect(res13_tst)
                    contract!(res13_tst,A,B,Val(1),Val(3))
                    @test collect(collect(res13)) ≈ collect(res13_tst)

                    # dimension 3, 4, and 5 should be equivalent
                    contract!(res13_tst,A,B,Val(1),Val(4))
                    @test collect(collect(res13)) ≈ collect(res13_tst)
                    contract!(res13_tst,A,B,Val(1),Val(5))
                    @test collect(collect(res13)) ≈ collect(res13_tst)

                    res16 = SymArray{(2,3),T}(arrType,N,N,N,N,N)
                    contract!(res16,A,S,Val(1),Val(6))
                    @tensor res16_tst[j,k,l,m,n] := A[i] * B[j,k,l,m,n,i]
                    @test collect(collect(res16)) ≈ collect(res16_tst)
                    contract!(res16_tst,A,B,Val(1),Val(6))
                    @test collect(collect(res16)) ≈ collect(res16_tst)

                    # check that contraction from SymArray to Array works
                    A = rand(T,M,O) |> arrType
                    S = SymArray{(1,2,1),T}(arrType,N,M,M,O)
                    rand!(S.data)
                    B = collect(collect(S)) |> arrType

                    res12 = zeros(T,O,N,M,O) |> arrType
                    res12_tst = zeros(T,O,N,M,O) |> arrType
                    contract!(res12,A,S,Val(1),Val(2))
                    contract!(res12_tst,A,B,Val(1),Val(2))
                    @test collect(res12) ≈ collect(res12_tst)
                end
            end
        end
    end
end