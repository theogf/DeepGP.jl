struct DeepGPModel{T<:Tuple,L}
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
Base.getindex(m::DeepGPModel,i::Int) = m.layers[1]


function ELBO(m::DeepGPModel,X,y)
    expec_log_like(m,X,y) - kl_divergence(m)
end

function kl_divergence(m::DeepGPModel)
    sum(kl_divergence(l) for l in m.layers)
end

propagate(m::DeepGPModel,f) = propagate(Base.tail(m.layers),propagate(first(m.layers),f))
propagate(::Tuple{}, f) = f
propagate(layers::Tuple,f) = propagate(Base.tail(layers),propagate(first(layers),f))

function expec_log_like(m,X,y,nSamples=1)
    @show f = propagate(m,[X for _ in 1:nSamples])
    # for l in m.layers
    #     f = propagate(l,f)
    # end
    loss = mean(_logpdf.(Ref(m),f,Ref(y)))
    return loss
end

function _logpdf(m::DeepGPModel,f,y)
    sum(_logpdf.(Ref(m.likelihood),f,y))
end

function _logpdf(l::Normal,f::Real,y::Real)
    Distributions.logpdf(l,f-y)
end
