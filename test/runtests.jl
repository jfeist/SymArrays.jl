using SymArrays
using Test

using SymArrays: symarrlength, sub2ind

@testset "SymArrays.jl" begin
    # Write your own tests here.
    @test symarrlength((3,6,4,3),(3,2,1,3)) == 8400

    S = SymArray{Float64,2,(2,)}(5,5);
    @test sub2ind(S,2,5) == sub2ind(S,5,2)
    @test sub2ind(S,3,4) != sub2ind(S,5,3)
    # sub2ind has to have same number of arguments as size of N
    @test_throws MethodError sub2ind(S,3,5,1)
    # indexing the array allows having additional 1s at the end
    @test S[3,5,1] == S[3,5]
    @test_throws BoundsError S[3,5,3]

    S = SymArray{Float64,8,(3,1,2,2)}(3,3,3,2,4,4,4,4);
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
    @test SymArray(A,(1,1)) == A
    @test SymArray(A,(2,)) != A

    A = A + A'
    # the standard equality test uses iteration over the arrays, (a,b) in zip(A,B),
    # which only accesses the actually different indices in SymArrays
    @test SymArray(A,(2,)) != A
    # broadcasting goes over all indices
    @test all(SymArray(A,(2,)) .== A)
end
