module DeepGP

using LinearAlgebra
using KernelFunctions
using Flux
using Zygote
using Distributions
using AugmentedGaussianProcesses
const AGP = AugmentedGaussianProcesses

include("layers.jl")
include("deepgp_base.jl")
include("training.jl")

end # module
