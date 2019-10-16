using Test
using SymArrays
using SymArrays: symarrlength, sub2ind
using TensorOperations
using Random

@testset "SymArrays.jl" begin
    @testset "SymArray" begin
        # Write your own tests here.
        @test symarrlength((3,6,4,3),(3,2,1,3)) == 8400

        S = SymArray{(2,),Float64}(5,5);
        @test sub2ind(S,2,5) == sub2ind(S,5,2)
        @test sub2ind(S,3,4) != sub2ind(S,5,3)
        # sub2ind has to have same number of arguments as size of N
        @test_throws MethodError sub2ind(S,3,5,1)
        # indexing the array allows having additional 1s at the end
        @test S[3,5,1] == S[3,5]
        @test_throws BoundsError S[3,5,3]

        S = SymArray{(3,1,2,2),Float64}(3,3,3,2,4,4,4,4);
        @test size(S) == (3,3,3,2,4,4,4,4)
        @test length(S) == 2000
        # iterating over all indices should give only the distinct indices,
        # i.e., give the same number of terms as the array length
        @test sum(1 for s in S) == length(S)
        @test sum(1 for I in eachindex(S)) == length(S)

        # calculating the linear index when iterating over Cartesian indices should give sequential access to the array
        @test 1:length(S) == [(sub2ind(S,Tuple(I)...) for I in eachindex(S))...]

        # test that permuting exchangeable indices accesses the same array element
        i1  = sub2ind(S,1,2,3,1,4,3,2,1)
        @test sub2ind(S,2,3,1,1,4,3,2,1) == i1
        @test sub2ind(S,2,3,1,1,3,4,2,1) == i1
        @test sub2ind(S,2,3,1,1,3,4,1,2) == i1
        @test sub2ind(S,2,1,3,1,4,3,1,2) == i1
        # make sure that swapping independent indices gives a different array element
        @test sub2ind(S,2,1,3,1,4,1,3,2) != i1
        @test sub2ind(S,1,2,3,2,4,3,2,1) != i1
        @test sub2ind(S,1,2,3,1,1,3,2,4) != i1

        SI = eachindex(S)
        @test first(SI) == CartesianIndex(1,1,1,1,1,1,1,1)

        A = rand(5,5)
        @assert A' != A
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

    @testset "Contractions" begin
        N,M,O = 10, 15, 18
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
end