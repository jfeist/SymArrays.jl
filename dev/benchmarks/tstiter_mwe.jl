using BenchmarkTools

println("Julia version $VERSION")

struct TstIter{M} n::Int end

function Base.iterate(iter::TstIter{M}) where M
    I = ntuple(i->1,Val(M))
    I, I
end
function Base.iterate(iter::TstIter, state)
    valid, I = __inc(iter, state)
    ifelse(valid, (I, I), nothing)
end

function __inc(iter::TstIter, state::NTuple{N,Int}) where {N}
    state[1] < state[2] && return true, (state[1]+1, Base.tail(state)...)
    valid, I = __inc(iter, Base.tail(state))
    return valid, (1, I...)
end
function __inc(iter::TstIter, state::Tuple{Int})
    state[1] < iter.n, (state[1]+1,)
end

function f(iter)
    s = 0
    for I in iter
        s += first(I)
    end
    s
end

n = 50
for M in 1:4
    iter = TstIter{M}(n)
    # check that number of elements is correct
    @assert count(x->true,iter) == binomial(n+M-1,M)

    print("M = $M: ")
    @btime f($iter)
end



function checkiter(iter::TstIter{M}) where M
    v = ones(Int,M)
    v[1] = 0
    for I in iter
        pos = 1
        v[pos] += 1
        while pos < M && v[pos] > v[pos+1]
            v[pos] = 1
            v[pos+1] += 1
            pos += 1
        end
        v[M] > iter.n && error("too many elements!")
        @assert Tuple(v) == I
    end
    @assert v == iter.n * ones(M)
end

for M in 1:4
    checkiter(TstIter{M}(n))
end