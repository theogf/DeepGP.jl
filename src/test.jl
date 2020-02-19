using MLDataUtils, LinearAlgebra
using DeepGP
using KernelFunctions
using Flux
using Zygote, ForwardDiff
using Distributions

X,y = noisy_sin(50,-1,1)
X = reshape(X,:,1)
X_test = reshape(collect(range(-1,1,length=200)),:,1)

##
ngp = 2; ndim = 15
μ1 = randn(ndim); Σ1 = diagm(ones(ndim));
μ2 = randn(ndim); Σ2 = diagm(ones(ndim));

Z = collect(reshape(range(-1,1,length=ndim),:,1))
l = 10.0
l1 = DeepGP.SVGPLayer(ngp,μ1,Σ1,Z,SqExponentialKernel(l),zeros(ndim))
l2 = DeepGP.SVGPLayer(1,μ2,Σ2,rand(ndim,ngp),SqExponentialKernel(l),zeros(ndim))
v1 = DeepGP.propagate(l1,X)
v2 = DeepGP.propagate(l2,v1)
m = DeepGP.DeepGPModel(l1,l2,likelihood=Normal(0,0.1))
DeepGP.expec_log_like(m,X,y)
DeepGP.params(m)
DeepGP.ELBO(m,X,y,2)
Zygote.refresh()
Zygote.gradient(x->DeepGP.ELBO(m,x,y),X)
DeepGP.mu(l1,X)
randn(size(X,1),l1.dim).*sqrt.(DeepGP.diag_sigma(l1,X))
T = 1000
opt = Momentum(0.0001)
y_pred = []
Zs = []
for i in 1:T
    global g = Zygote.gradient(()->DeepGP.ELBO(m,X,y),Params(DeepGP.params(m)))
    for l in m
        for gp in l
            gp.μ .+= Flux.Optimise.apply!(opt,gp.μ,g[gp][].μ)
            gp.L .+= Flux.Optimise.apply!(opt,gp.L,LowerTriangular(g[gp][].L))
            # gp.Z .+= Flux.Optimise.apply!(opt,gp.Z,g[gp][].Z)
        end
    end
    push!(y_pred,DeepGP.predict(m,X_test,100))
    push!(Zs,getproperty.(m[1],:Z))
    display(DeepGP.ELBO(m,X,y))
end

##
using Plots; pyplot()
histogram([  for _ in 1:1000]) |> display

scatter(X,y,lab="",markeralpha=0.5,markerstrokewidth=0.0)
y_pred[1]
anim = @animate for i in 1:1:length(y_pred)
    scatter(X,y,lab="",markeralpha=0.5,markerstrokewidth=0.0,ylims=(-3,3))
    scatter!(Zs[i][1],zeros(ndim),label="")
    plot!(X_test,mean(y_pred[i]),lab="",title="Iteration = $i",ribbon=sqrt.(diag(cov(y_pred[i]))),fillalpha=0.3)
end

gif(anim,fps=5)
