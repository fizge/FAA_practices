

# Archivo de pruebas para realizar autoevaluación de algunas funciones de los ejercicios

# Importamos el archivo con las soluciones a los ejercicios
include("soluciones.jl");
#   Cambiar "soluciones.jl" por el nombre del archivo que contenga las funciones a desarrollar

# Fichero de pruebas realizado con la versión 1.11.2 de Julia
println(VERSION)
#  y la 1.11.2 de Random
println(Random.VERSION)
#  y la versión 0.16.0 de Flux
import Pkg
Pkg.status("Flux")

# Es posible que con otras versiones los resultados sean distintos, estando las funciones bien, sobre todo en la funciones que implican alguna componente aleatoria
# Cargamos el dataset
using DelimitedFiles: readdlm
dataset = readdlm("iris.data",',');
# Preparamos las entradas
inputs = convert(Array{Float32,2}, dataset[:,1:4]);


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------


# Establecemos la semilla para que los resultados sean siempre los mismos
using Random: seed!
# Comprobamos que la generación de números aleatorios es la esperada:
seed!(1); @assert(isapprox(rand(), 0.07336635446929285))
#  Si fallase aquí, seguramente dara error al comprobar los resultados de la ejecución de la siguiente función porque depende de la generación de números aleatorios

# Comprobamos la función trainClassANN con estos datos
#  Como se puede ver, los conjuntos de entrenamiento, validacion y test se solapan. Esto no es correcto, pero se hace para forzar al entrenamiento a que se pare antes de tiempo por validación (parada temprana)
seed!(1); (ann, trainingLosses, validationLosses, testLosses) = trainClassANN([4,3], (inputs, targets);
    validationDataset=(inputs[101:150,:], targets[101:150,:]),
    testDataset=(inputs[51:100,:], targets[51:100,:]),
    maxEpochs=100, maxEpochsVal=5); length(trainingLosses)
# Los vectores de loss que debería devolver son los siguientes:
result_trainingLosses   = Float32[1.2139437, 1.2000579, 1.1873707, 1.1757828, 1.165133,  1.155307, 1.1462542, 1.1379561, 1.1304016, 1.1235778, 1.117466,  1.1120408];
result_validationLosses = Float32[1.1530712, 1.1259663, 1.1016681, 1.0822797, 1.0691929, 1.0617257, 1.0585984, 1.0587119, 1.061241, 1.065562,  1.0711765, 1.0776592];
result_testLosses       = Float32[1.8724904, 1.8293855, 1.787262,  1.7450062, 1.7017598, 1.6577507, 1.6135957, 1.569862, 1.5269873, 1.4853015, 1.4450579, 1.4064597];
@assert(all(isequal.(trainingLosses,   result_trainingLosses)))
@assert(all(isequal.(validationLosses, result_validationLosses)))
@assert(all(isequal.(testLosses,       result_testLosses)))


function holdOut(N::Int, P::Float64)
    @assert 0 <= P <= 1 "P debe estar entre 0 y 1"
    indices = randperm(N)  # Generate a random permutation of indexes
    split_point = floor(Int, N * (1 - P))  # Determines cut-off point
    return indices[1:split_point], indices[split_point+1:end]  # Returns the tuple with the subdatasets
end


function holdOut(N::Int, Pval::Float64, Ptest::Float64)
    @assert 0 <= Pval + Ptest <= 1 "La suma de Pval y Ptest debe estar entre 0 y 1"
    train_test, test = holdOut(N, Ptest)  # Separates the test dataset
    Pval_adjusted = Pval / (1 - Ptest)  # Adjusts Pval for the remaining datasets 
    train, val = holdOut(length(train_test), Pval_adjusted)  # Separates validate dataset from training dataset
    return train_test[train], train_test[val], test  # Returns the three datasets
end