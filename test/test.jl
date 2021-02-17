using DeepGP
using MLDataUtils, LinearAlgebra
using KernelFunctions
using Distributions

X, y = noisy_sin(50, -1, 1)
X = reshape(X, :, 1)
X_test = reshape(collect(range(-1, 1, length=200)), :, 1)

##
ngp = 10; ndim = 15
μ1 = randn(ndim); L1 = LowerTriangular(diagm(ones(ndim)));
μ2 = randn(ndim); L2 = LowerTriangular(diagm(ones(ndim)));

Z = collect(reshape(range(-1, 1, length=ndim),:,1))
l = 10.0
m1 = DeepGP.LinearMean(rand(1, 1))
m2 = DeepGP.LinearMean(rand(ngp, 1))
l1 = DeepGP.SVGPLayer(ngp, μ1, L1, Z, transform(SqExponentialKernel(), l), m1)
l2 = DeepGP.SVGPLayer(1, μ2, L2, rand(ndim, ngp), transform(SqExponentialKernel(), l), m2)
v1 = DeepGP.propagate(l1, X)
v2 = DeepGP.propagate(l2, v1)
m = DeepGP.DeepGPModel(l1, l2,likelihood=Normal(0,0.1))
DeepGP.expec_log_like(m, X, y)
DeepGP.params(m)
DeepGP.ELBO(m, X, y, nsamples=2)

using Flux
using Zygote
xg = Zygote.gradient(x->DeepGP.ELBO(m, x, y), X)
mean(l1, X)
randn(size(X, 1), l1.dim) .* sqrt.(DeepGP.var(l1, X))
T = 1000
opt = Momentum(0.01)
y_pred = []
Zs = []
Zygote.Params(m)
ps = Params(DeepGP.params(m))
for i in 1:10#T
    global g = Zygote.gradient(()->DeepGP.ELBO(m, X, y), ps)
    for l in m
        for gp in l
            gp.μ .+= Flux.Optimise.apply!(opt, gp.μ, g[gp][].μ)
            gp.L .+= LowerTriangular(Flux.Optimise.apply!(opt, gp.L, LowerTriangular(g[gp][].L)))
            if gp.μ₀ isa DeepGP.LinearMean
                gp.μ₀.A .+= Flux.Optimise.apply!(opt, gp.μ₀.A, g[gp][].μ₀[1])
            end
            # gp.Z .+= Flux.Optimise.apply!(opt,gp.Z,g[gp][].Z)
        end
    end
    push!(y_pred, DeepGP.predict(m, X_test, nsamples=100))
    push!(Zs, getproperty.(m[1], :Z))
    display(DeepGP.ELBO(m, X, y))
end
DeepGP.propagate(m[1], X)
##
using Plots; pyplot()
# histogram([  for _ in 1:1000]) |> display

scatter(X, y, lab="", markeralpha=0.5, markerstrokewidth=0.0)
y_pred[1]
anim = @animate for i in 1:1:length(y_pred)
    scatter(X, y, lab="", markeralpha=0.5, markerstrokewidth=0.0, ylims=(-3,3))
    # scatter!(Zs[i][1], zeros(ndim),label="")
    plot!(X_test, mean(y_pred[i]), lab="",title="Iteration = $i", ribbon=sqrt.(diag(cov(y_pred[i]))),fillalpha=0.3)
end

gif(anim,fps=5)
