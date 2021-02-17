module DeepGP

using Distributions
using Functors
using KernelFunctions
using LinearAlgebra
using PDMats
using Statistics

include("utils.jl")
include("layers.jl")
include("deepgp_base.jl")
include("training.jl")
end # module
