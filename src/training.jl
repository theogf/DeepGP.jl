function train!(model,X,y,opt=Adam())
    # ps = Flux.params(model)
    loss(X,y) = ELBO(model,X,y)
    # Flux.train!(loss,ps,(X,y),opt)
end
