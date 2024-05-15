include("cfb/cfb.jl")
using Main.CFB, BenchmarkTools

# Escolha do arquivo a ser utilizado
sistema = "107barras"
arquivo = "107barras.txt"


# Chamadas ao cálculo da CFB
println("------ 107barras -------")
g = CFB.read_edgelist(arquivo)
@time CFB.cfb_de(g, 1, 100, 0.7, 0.2, 0.8, 10, sistema)
@time CFB.cfb_de(g, 2, 100, 0.7, 0.2, 0.8, 10, sistema)
@time CFB.cfb_de(g, 3, 100, 0.7, 0.2, 0.8, 10, sistema)