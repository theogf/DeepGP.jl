abstract type AbstractGPLayer end;

const jitt = 1e-5

struct SVGPLayer{GPType} <: AbstractGPLayer
    dim::Int
    gps::Vector{GPType}
    function SVGPLayer(dim, μ, L, Z, kernel, μ₀)
        gps = [
            SVGPBase(copy(μ), copy(L), copy(Z), deepcopy(kernel), deepcopy(μ₀)) for
            _ in 1:dim
        ]
        return new{SVGPBase}(dim, gps)
    end
end

params(l::SVGPLayer) = params.(l.gps)

Base.length(l::SVGPLayer) = l.dim
Base.getindex(l::SVGPLayer, i::Int) = l.gps[i]
Base.iterate(l::SVGPLayer) = iterate(l.gps)
Base.iterate(l::SVGPLayer, state::Any) = iterate(l.gps, state)

function propagate(l::AbstractGPLayer, samples::AbstractVector{<:AbstractArray})
    compute_K.(l.gps)
    return propagate.(Ref(l), samples)
end

## Return the parametrization m + eps * sqrt(S)
function propagate(l::AbstractGPLayer, samples::AbstractArray{<:Real})
    return mean(l, samples) .+ randn(size(samples, 1), l.dim) .* sqrt.(var(l, samples))
end

# mu(l::AbstractGPLayer,x) = reduce(hcat,mu(gp,x) for gp in l.gps)
function Distributions.mean(l::AbstractGPLayer, x::AbstractArray)
    return reduce(hcat, mean(gp, x) for gp in l)
end
Distributions.var(l::AbstractGPLayer, x) = reduce(hcat, var(gp, x) for gp in l)
sigma(l::AbstractGPLayer, x) = sigma.(l, Ref(x))

KL(l::AbstractGPLayer) = sum(KL, l.gps)

abstract type AbstractGPBase end;

mutable struct SVGPBase{T,TK} <: AbstractGPBase
    dim::Int
    μ::Vector{T}
    L::LowerTriangular{T,Matrix{T}}
    μ₀::PriorMean
    kernel::Kernel
    Z::Matrix
    K::TK
    function SVGPBase(
        dim, Z::Matrix{T}, kernel=SqExponentialKernel(), μ₀::PriorMean=ZeroMean(); usepdmat=false
    ) where {T}#AGP.ZeroMean()) where {T}
        K = usepdmat ? PDMat(diagm(ones(T, dim))) : Symmetric(diagm(ones(T, dim)))
        return new{T,typeof(K)}(
            dim, zeros(T, dim), LowerTriangular(diagm(ones(T, dim))), μ₀, kernel, Z, K
        )
    end
    function SVGPBase(
        μ::Vector{T},
        L::LowerTriangular{T},
        Z::Matrix,
        kernel::KernelFunctions.Kernel,
        μ₀::PriorMean=ZeroMean(),
        usepdmat=false,
    ) where {T} #,μ₀::AGP.PriorMean) where {T}
        d = length(μ)
        K = usepdmat ? PDMat(diagm(ones(T, d))) : Symmetric(diagm(ones(T, d)))
        return new{T,typeof(K)}(d, μ, L, μ₀, kernel, Z, K)
    end
end

@functor SVGPBase

params(gp::SVGPBase) = (gp.μ, gp.L, gp.Z)

function compute_K(gp::SVGPBase{T,<:Symmetric}) where {T}
    return gp.K = Symmetric(kernelmatrix(kernel(gp), gp.Z; obsdim=1) + jitt * I)
end

function compute_K(gp::SVGPBase{T,<:PDMat}) where {T}
    return gp.K = kernelpdmat(kernel(gp), gp.Z; obsdim=1)
end

"""
    κ(gp, X)

Return the κ matrix defined as KₓᵤKᵤᵤ⁻¹ 
"""
function κ(gp::SVGPBase, X::AbstractArray)
    ## Return the K_XZ K
    return kernelmatrix(kernel(gp), X, gp.Z; obsdim=1) / gp.K
end

kernel(gp::AbstractGPBase) = gp.kernel

function KernelFunctions.kernelmatrix(gp::AbstractGPBase, x)
    return kernelmatrix(kernel(gp), x; obsdim=1) + jitt * I
end
function KernelFunctions.kernelmatrix(gp::AbstractGPBase, x, y)
    return kernelmatrix(kernel(gp), x, y; obsdim=1)
end
function KernelFunctions.kernelmatrix_diag(gp::AbstractGPBase, x)
    return kernelmatrix_diag(kernel(gp), x; obsdim=1) .+ jitt
end

# mu(gp::SVGP_Base,x)= gp.μ₀(x) + κ(gp,x,gp.invKmm)*(gp.μ) -gp.μ₀(gp.Z)
Distributions.mean(gp::SVGPBase) = gp.μ
Distributions.mean(gp::SVGPBase, x::AbstractArray) = κ(gp, x) * (gp.μ - gp.μ₀(gp.Z))
Distributions.var(gp::SVGPBase) = opt_diag(gp.L)
Distributions.cov(gp::SVGPBase) = XXt(gp.L)
covprior(gp::SVGPBase) = gp.K
function Distributions.var(gp::SVGPBase, x, κ=κ(gp, x))
    return diag(kernelmatrix(gp, x)) - opt_diag(κ * (gp.K - cov(gp)), κ)
end
function Distributions.cov(gp::SVGPBase, x, κ=κ(gp, x, gp.K))
    return kernelmatrix(gp, x) - κ * (covprior(gp) - cov(gp)) * κ'
end

function KL(gp::SVGPBase)
    return 0.5 * (
        -logdet(cov(gp)) +
        logdet(gp.K) +
        opt_trace(inv(covprior(gp)), cov(gp)) +
        invquad(covprior(gp), mean(gp) - gp.μ₀(gp.Z)) - length(gp.μ)
    )
end #WARNING removed μ₀
