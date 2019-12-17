module SymArrays

export SymArray, contract, contract!, SymArr_ifsym

include("helpers.jl")

include("symarray.jl")

include("contractions.jl")

using CUDAapi
if has_cuda_gpu()
    include("cuda_contractions.jl")
end

end # module
