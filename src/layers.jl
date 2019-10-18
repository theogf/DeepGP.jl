abstract type AbstractGPLayer end;

struct SVGPLayer{GPType} <: AbstractGPLayer
    dim::Int
    gps::Vector{GPType}
    function SVGPLayer(dim,μ,Σ,Z,kernel,μ₀)
        gps = [SVGP_Base(copy(μ),copy(Σ),copy(Z),deepcopy(kernel),deepcopy(μ₀)) for _ in 1:dim]
        new{SVGP_Base}(dim,gps)
    end
end

Base.length(l::SVGPLayer) = l.dim
Base.getindex(l::SVGPLayer,i::Int) = l.gps[i]

function propagate(l::AbstractGPLayer,samples::AbstractVector{<:AbstractArray})
    compute_invK.(l.gps)
    propagate.([l],samples)
end
function propagate(l::AbstractGPLayer,samples::AbstractArray{<:Real})
    ## Return the parametrization m + eps *sqrt(S)
    mu(l,samples) + randn(size(samples,1),l.dim).*sqrt.(diag_sigma(l,samples))
end

mu(l::AbstractGPLayer,x) = reduce(hcat,[mu(gp,x) for gp in l.gps])
diag_sigma(l::AbstractGPLayer,x) = reduce(hcat,[diag_sigma(gp,x) for gp in l.gps])

kl_divergence(l::AbstractGPLayer) = sum(kl_divergence(gp) for gp in l.gps)

abstract type AbstractGPBase end;

mutable struct SVGP_Base{T} <: AbstractGPBase
    dim::Int
    μ::Vector{T}
    Σ::Matrix{T}
    μ₀::AGP.PriorMean
    kernel::KernelFunctions.Kernel
    Z::Matrix
    invKmm::Matrix{T}
    function SVGP_Base(dim,Z::Matrix{T},kernel=SqExponentialKernel(),μ₀=ZeroMean()) where {T}
        new{T}(dim,zeros(T,dim),diagm(ones(T,dim)),μ₀,kernel,Z,diagm(ones(T,dim)))
    end
    function SVGP_Base(μ::Vector{T},Σ::Matrix{T},Z::Matrix,kernel::KernelFunctions.Kernel,μ₀::AGP.PriorMean) where {T}
        new{T}(length(μ),μ,Σ,μ₀,kernel,Z,diagm(ones(T,length(μ))))
    end
end

function compute_invK(gp::SVGP_Base)
    gp.invKmm = inv(KernelFunctions.kernelmatrix(gp.kernel,gp.Z,obsdim=1)+1e-5I)
end

function κ(gp::SVGP_Base,X::AbstractArray,invKmm)
    ## Return the K_XZ K
    try
        KernelFunctions.kernelmatrix(gp.kernel,X,gp.Z,obsdim=1)*invKmm
    catch e
        @show size(X)
        @show size(gp.Z)
        rethrow(e)
    end
end

_k(gp::AbstractGPBase,x,y) = KernelFunctions.kernelmatrix(gp.kernel,x,y,obsdim=1)
_diagk(gp::AbstractGPBase,x) = KernelFunctions.kerneldiagmatrix(gp.kernel,x,obsdim=1).+1e-5

mu(gp::SVGP_Base,x)= gp.μ₀ + κ(gp,x,gp.invKmm)*(gp.μ-gp.μ₀)
diag_sigma(gp::SVGP_Base,x) = diag_sigma(gp,x,κ(gp,x,gp.invKmm))
diag_sigma(gp::SVGP_Base,x,κ) = _diagk(gp,x) - AGP.opt_diag(κ*(gp.invKmm-gp.Σ),κ)

kl_divergence(gp::SVGP_Base) = 0.5*(-logdet(gp.Σ)-logdet(gp.invKmm)+AGP.opt_trace(gp.invKmm,gp.Σ)+dot(gp.μ-gp.μ₀,gp.invKmm*(gp.μ-gp.μ₀))-length(gp.μ))
