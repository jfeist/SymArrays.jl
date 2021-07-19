"""based on Base.binomial, but without negative values for n and without overflow checks
(index calculations here should not overflow if the array does not have more elements than an Int64 can represent)"""
@inline function binomial_simple(n::T, k::T) where T<:Integer
    (k < 0 || k > n) && return zero(T)
    (k == 0 || k == n) && return one(T)
    if k > (n>>1)
        k = n - k
    end
    k == 1 && return n
    x::T = nn = n - k + 1
    nn += 1
    rr = 2
    while rr <= k
        x = div(x*nn, rr)
        rr += 1
        nn += 1
    end
    x
end

@inline @generated function binomial_unrolled(n::T, ::Val{k}) where {k,T<:Integer}
    terms = [:(n - $(k-j)) for j=1:k]
    if k < 10
        # for small k, just return the unrolled calculation directly
        # (n k) = prod(n+i-k, i=1:k)/k!
        # typemax(Int)/factorial(9) ~ 23*10^12
        # so for k<10, this does not overflow for index calculation of up to ~100TB arrays.
        # so we do not worry about it
        # use the precomputed factorial
        binom = :( *($(terms...)) ÷ $(factorial(k)) )
    else
        binom = terms[1] # j=1
        for j=2:k
            # careful about operation order:
            # first multiply, the product is then always divisible by j
            binom = :( ($binom * $(terms[j])) ÷ $j )
        end
        # (n k) == (n n-k) -> should be faster for n-k < k
        # but when we do this replacement, we cannot unroll the loop explicitly,
        # so heuristically use n-k < k/2 to ensure that it wins
        :( n < $(3k÷2) ? binomial_simple(n,n-$k) : $binom )
    end
end

"""calculate binomial(ii+n+offset,n)
This shows up in size and index calculations for arrays with symmetric indices."""
#@inline symind_binomial(ii,::Val{n},::Val{offset}) where {n,offset} = binomial_unrolled(ii+(n+offset),Val(n))

# binary search for finding an integer m such that func(m) <= ind < func(m+1), with low <= m <= high
function searchlast_func(x,func,low::T,high::T) where T<:Integer
    high += one(T)
    while low < high-one(T)
        mid = (low+high) >> 1
        if x < func(mid)
            high = mid
        else
            low = mid
        end
    end
    return low
end

"""
    sort_track_parity(t::Tuple) -> ::Int, ::Tuple

Sorts the tuple `t` and tracks whether the permutation needed to sort it is even
or odd. Returns (p, ts), where p = ±1 and ts is the sorted tuple. Algorithm
copied from TupleTools (removing optional lt, by, rev, but adding the
tracking of the permutation sign).
"""
@inline function sort_track_parity(t::Tuple)
    t1, t2 = TupleTools._split(t)
    p1,t1s = sort_track_parity(t1)
    p2,t2s = sort_track_parity(t2)
    pm,ts = _merge_track_parity(t1s, t2s)
    return pm*p1*p2, ts
end
@inline sort_track_parity(t::Tuple{Any}) = 1, t
@inline sort_track_parity(t::Tuple{}) = 1, t

function _merge_track_parity(t1::Tuple, t2::Tuple)
    if first(t1) < first(t2)
        pm, tm = _merge_track_parity(Base.tail(t1), t2)
        return pm, (first(t1), tm...)
    else
        pm, tm = _merge_track_parity(t1, Base.tail(t2))
        # if first(t1)==first(t2), order and thus parity are not well defined
        # otherwise, there are length(t1) inversions
        pm = first(t1) == first(t2) ? 0 : (iseven(length(t1)) ? pm : -pm)
        return pm, (first(t2), tm...)
    end
end
_merge_track_parity(::Tuple{}, t2::Tuple) = 1, t2
_merge_track_parity(t1::Tuple, ::Tuple{}) = 1, t1
_merge_track_parity(::Tuple{}, ::Tuple{}) = 1, ()
