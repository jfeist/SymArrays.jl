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
    assets=String[],
)

deploydocs(;
    repo="github.com/jfeist/SymArrays.jl",
)
