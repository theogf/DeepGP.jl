abstract type PriorMean end

struct LinearMean{T<:Real,A<:AbstractMatrix{T}} <: PriorMean
    A::A
end

(μ::LinearMean)(x::AbstractMatrix) = vec(x * μ.A)
(μ::LinearMean)(x::AbstractVector) =  dot.(x, Ref(μ.A))

params(m::LinearMean) = m.A

@functor LinearMean

struct ZeroMean{T<:Real} <: PriorMean end

ZeroMean() = ZeroMean{Float64}()

(μ::ZeroMean{T})(x::AbstractMatrix) where {T} = zeros(T, size(x, 1))
(μ::ZeroMean{T})(x::AbstractVector) where {T} = zeros(T, length(x))




@inline function opt_diag(A::AbstractArray, B::AbstractArray=A)
    return vec(sum(A .* B; dims=2))
end

@inline function opt_trace(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})
    return dot(A, B)
end

XXt(X) = X * X'

PDMats.invquad(a::Symmetric, b::AbstractVector) = dot(b, a \ b)