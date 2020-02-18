abstract type AbstractGPLayer end;

const jitt = 1e-3

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
    compute_K.(l.gps)
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
    μ₀::Vector#AGP.PriorMean
    kernel::KernelFunctions.Kernel
    Z::Matrix
    K::Symmetric{T,Matrix{T}}
    function SVGP_Base(dim,Z::Matrix{T},kernel=SqExponentialKernel(),μ₀=zeros(dim)) where {T}#AGP.ZeroMean()) where {T}
        new{T}(dim,zeros(T,dim),diagm(ones(T,dim)),μ₀,kernel,Z,Symmetric(diagm(ones(T,dim))))
    end
    function SVGP_Base(μ::Vector{T},Σ::Matrix{T},Z::Matrix,kernel::KernelFunctions.Kernel,μ₀=zero(μ)) where {T} #,μ₀::AGP.PriorMean) where {T}
        new{T}(length(μ),μ,Σ,μ₀,kernel,Z,Symmetric(diagm(ones(T,length(μ)))))
    end
end

function compute_K(gp::SVGP_Base)
    gp.K = Symmetric(KernelFunctions.kernelmatrix(gp.kernel,gp.Z,obsdim=1)+jitt*I)
end

function κ(gp::SVGP_Base,X::AbstractArray,K)
    ## Return the K_XZ K
    KernelFunctions.kernelmatrix(gp.kernel,X,gp.Z,obsdim=1)/K
end

_k(gp::AbstractGPBase,x,y) = KernelFunctions.kernelmatrix(gp.kernel,x,y,obsdim=1)
_diagk(gp::AbstractGPBase,x) = KernelFunctions.kerneldiagmatrix(gp.kernel,x,obsdim=1).+jitt

# mu(gp::SVGP_Base,x)= gp.μ₀(x) + κ(gp,x,gp.invKmm)*(gp.μ) -gp.μ₀(gp.Z)
mu(gp::SVGP_Base,x)= κ(gp,x,gp.K)*(gp.μ) #-gp.μ₀(x)
diag_sigma(gp::SVGP_Base,x) = diag_sigma(gp,x,κ(gp,x,gp.K))
diag_sigma(gp::SVGP_Base,x,κ) = _diagk(gp,x) - opt_diag(κ*(gp.K-gp.Σ),κ)

kl_divergence(gp::SVGP_Base) = 0.5*(-logdet(gp.Σ)+logdet(gp.K) + opt_trace(inv(gp.K),gp.Σ)+dot(gp.μ,gp.K\(gp.μ))-length(gp.μ)) #WARNING removed μ₀


@inline function opt_diag(A::AbstractArray{T,N},B::AbstractArray{T,N}) where {T<:Real,N}
    vec(sum(A.*B,dims=2))
end

@inline function opt_trace(A::AbstractMatrix{<:Real},B::AbstractMatrix{<:Real})
    dot(A,B)
end
