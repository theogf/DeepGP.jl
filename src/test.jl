using MLDataUtils, LinearAlgebra
using DeepGP
using AugmentedGaussianProcesses
using KernelFunctions
using Flux
using Zygote
using Distributions

X,y = noisy_sin(100,-1,1)
X = reshape(X,:,1)


ngp = 1; ndim = 15
μ = zeros(ndim); Σ = diagm(ones(ndim));
Z = collect(reshape(range(-1,1,length=ndim),:,1))
l = 10.0
l1 = DeepGP.SVGPLayer(ngp,μ,Σ,Z,SqExponentialKernel(l),ZeroMean())
l2 = DeepGP.SVGPLayer(1,μ,Σ,rand(ndim,ngp),SqExponentialKernel(l),ZeroMean())
v1 = DeepGP.propagate(l1,X)
v2 = DeepGP.propagate(l2,v1)
m = DeepGP.DeepGPModel(l1)
DeepGP.expec_log_like(m,X,y)
DeepGP.ELBO(m,X,y) |> display

Zygote.refresh()
Zygote.gradient(x->DeepGP.ELBO(m,x,y),X)
[X for _ in 1:nSamples]
t
