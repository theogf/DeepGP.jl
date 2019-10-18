struct GPLayer <: AbstractGPLayer
    dim::Int
    μ₀::Vector{MeanPrior}
    kernel::Vector{Kernel}
    Z::Vector{AbstractMatrix}
    Kmm::Vector{AbstractPDMats}
end

function transform(l::GPLayer,sample::AbstractVector)
    ## Return the parametrization m + eps *sqrt(S)
end

function κ(l::GPLayer,X::AbstractVector,Z::AbstractMatrix)
    ## Return the K_XZ K
    invKmm = inv(kernelmatrix(l.kernel,Z))
    kernelmatrix(X,Z,invKmm)
end

function κ(l::GPLayer,X::AbstractVector,Z::AbstractMatrix,invKmm)
    ## Return the K_XZ K
    kernelmatrix(l.kernel,X)*invKmm
end

function mu(l::GPLayer)

end

function sig(l::GPLayer)
