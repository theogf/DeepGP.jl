@adjoint Base.reduce(::typeof(hcat), V::AbstractVector{<:AbstractVector}) =
    reduce(hcat, V), dV -> (nothing, collect(eachcol(dV)),)
