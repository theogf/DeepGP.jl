struct DeepGPModel
    X::AbstractMatrix
    y::AbstractVector
    layers::Vector{AbstractGPLayer}
    nLayers::Int
    likelihood::Likelihood
end

function ELBO(m::DeepGP,y)
    expec_log_like(m,y) - kl_divergence(m)
end

function kl_divergence(m)

end

function expec_log_like(m,y)

end
