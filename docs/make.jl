using SymArrays
using Documenter

DocMeta.setdocmeta!(SymArrays, :DocTestSetup, :(using SymArrays); recursive=true)

makedocs(;
    modules=[SymArrays],
    authors="Johannes Feist <johannes.feist@gmail.com> and contributors",
    repo="https://github.com/jfeist/SymArrays.jl/blob/{commit}{path}#{line}",
    sitename="SymArrays.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jfeist.github.io/SymArrays.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jfeist/SymArrays.jl",
    devbranch = "main",
)
