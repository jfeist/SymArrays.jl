using Documenter, SymArrays

makedocs(;
    modules=[SymArrays],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/jfeist/SymArrays.jl/blob/{commit}{path}#L{line}",
    sitename="SymArrays.jl",
    authors="Johannes Feist",
)

deploydocs(;
    repo="github.com/jfeist/SymArrays.jl.git",
)
