using MLDataUtils, LinearAlgebra
using DeepGP
using KernelFunctions
using Flux
using Zygote, ForwardDiff
using Distributions

X,y = noisy_sin(50,-1,1)
X = reshape(X,:,1)


ngp = 1; ndim = 15
μ1 = randn(ndim); Σ1 = diagm(ones(ndim));
μ2 = randn(ndim); Σ2 = diagm(ones(ndim));

Z = collect(reshape(range(-1,1,length=ndim),:,1))
l = 10.0
l1 = DeepGP.SVGPLayer(ngp,μ1,Σ1,Z,SqExponentialKernel(l),zeros(ndim))
l2 = DeepGP.SVGPLayer(1,μ2,Σ2,rand(ndim,ngp),SqExponentialKernel(l),zeros(ndim))
v1 = DeepGP.propagate(l1,X)
v2 = DeepGP.propagate(l2,v1)
m = DeepGP.DeepGPModel(l1,l2)
DeepGP.expec_log_like(m,X,y)
DeepGP.ELBO(m,X,y) |> display

Zygote.refresh()
Zygote.gradient(x->DeepGP.ELBO(m,x,y),X)
Zygote.gradient(_->DeepGP.ELBO(m,X,y),Params([μ1]))
ForwardDiff.gradient(x->DeepGP.ELBO(m,x,y),X)
[X for _ in 1:nSamples]
t
