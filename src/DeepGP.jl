module DeepGP

using LinearAlgebra
using KernelFunctions
using ZygoteRules
using Distributions

include("layers.jl")
include("deepgp_base.jl")
include("training.jl")

end # module
