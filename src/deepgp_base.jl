struct DeepGPModel{T<:Tuple,L}
    layers::T
    nLayers::Int
    likelihood::L
end

function DeepGPModel(layers...;likelihood=AGP.GaussianLikelihood())
    # Here do some checks with likelihood and last layer
    # Check dimensions between layers
    DeepGPModel{typeof(layers)}(layers,length(layers),likelihood)
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
propagate(layers::Tuple,f) = propagate(Base.tail(layers),propagate(first(layer),f))

function expec_log_like(m,X,y,nSamples=1)
    global f = [X for _ in 1:nSamples]
    propagate(m,f)
    for l in m.layers
        f = propagate(l,f)
    end
    loss = 0.0
    for i in 1:nSamples
        loss += logpdf(m,f[i],y)/nSamples
    end
    return loss
end

function logpdf(m::DeepGPModel,f,y)
    tot = 0.0
    for i in 1:length(y)
        tot += AGP.logpdf(m.likelihood,f[i],y[i]) #Would not work for multiclass for example
    end
    return tot
end
