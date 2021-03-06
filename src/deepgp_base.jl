struct DeepGPModel{T,L}
    layers::T
    nLayers::Int
    likelihood::L
end

function DeepGPModel(layers...;likelihood=Normal())
    # Here do some checks with likelihood and last layer
    # Check dimensions between layers
    DeepGPModel{typeof(layers),typeof(likelihood)}(layers,length(layers),likelihood)
end

# Base.:∘(l1::AbstractGPLayer,l2::AbstractGPLayer) = DeepGPModel([l1,l2])
Base.getindex(m::DeepGPModel, i::Int) = m.layers[i]
Base.length(m::DeepGPModel) = m.length
Base.iterate(m::DeepGPModel) = iterate(m.layers)
Base.iterate(m::DeepGPModel, state) = iterate(m.layers, state)

params(m::DeepGPModel) = params.(m.layers) #Eventually add parameters from likelihood

function ELBO(m::DeepGPModel, X, y; nsamples=1)
    expec_log_like(m, X, y, nsamples=nsamples) - KL(m)
end

function KL(m::DeepGPModel)
    sum(KL, m)
end

propagate(m::DeepGPModel, f) = propagate(Base.tail(m.layers), propagate(first(m), f))
propagate(::Tuple{}, f) = f
propagate(layers::Tuple, f) = propagate(Base.tail(layers), propagate(first(layers), f))

"""
    predict(m, X; nsamples)

Create `nsamples` samples with model `m` given samples `X`
"""
function predict(m::DeepGPModel, X; nsamples=1)
    propagate(m, [X for _ in 1:nsamples])
end

function expec_log_like(m, X, y; nsamples=1)
    f = predict(m, X, nsamples=nsamples)
    loss = mean(_logpdf.(Ref(m), f, Ref(y)))
    return loss
end

function _logpdf(m::DeepGPModel, f, y)
    sum(_logpdf.(Ref(m.likelihood), f, y))
end

function _logpdf(l::Normal, f::Real, y::Real)
    -0.5 * abs2(y - f) / l.σ^2
    # Distributions.logpdf(l,f-y)
end
