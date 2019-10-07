using Documenter, DeepGP

makedocs(;
    modules=[DeepGP],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/theogf/DeepGP.jl/blob/{commit}{path}#L{line}",
    sitename="DeepGP.jl",
    authors="Theo Galy-Fajou",
    assets=String[],
)

deploydocs(;
    repo="github.com/theogf/DeepGP.jl",
)
