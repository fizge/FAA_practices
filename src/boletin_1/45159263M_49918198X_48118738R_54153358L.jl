using DelimitedFiles
using Statistics
using Flux
using Flux.Losses

## One Hot Encoding

function oneHotEncoding(feature:: AbstractArray{<:Any, 1}, classes:: AbstractArray{<:Any, 1})

    num_classes = length(classes)

    if num_classes <= 2
        oneHot = Matrix{Bool}(undef, length(feature), 1)
        oneHot[:, 1] = feature .== classes[1]
    else
        oneHot = Matrix{Bool}(undef, length(feature), num_classes)
        for i in 1:num_classes
            oneHot[:, i] = feature .== classes[i]
        end
    end

    return oneHot
end

function oneHotEncoding(feature:: AbstractArray{<:Any, 1}) 
    
    return oneHotEncoding(feature, unique(feature))
end

function oneHotEncoding(feature:: AbstractArray{Bool, 1}) 
    
    return reshape(feature, :, 1)
end

# Calcular parámetros de normalización

function calculateMinMaxNormalizationParameters(dataset:: AbstractArray{<:Real, 2})
    
    mins = minimum(dataset, dims=1)
    maxs = maximum(dataset, dims=1)
    return (mins, maxs)
end

function calculateZeroMeanNormalizationParameters(dataset:: AbstractArray{<:Real, 2})
    
    meanValues = mean(dataset, dims=1)
    stdValues = std(dataset, dims=1)
    return (meanValues, stdValues)
end

# Funciones para normalizar dataset

function normalizeMinMax!(dataset:: AbstractArray{<: Real, 2}, normalizationParameters:: NTuple{2, AbstractArray{<: Real, 2}})
    
    mins, maxs = normalizationParameters
    dataset .-= mins
    dataset ./= (maxs .- mins)
    dataset[:, vec(mins.==maxs)] .= 0
    return dataset

end

function normalizeMinMax!(dataset:: AbstractArray{<: Real, 2})
    
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    return normalizeMinMax!(dataset, normalizationParameters)
end

function normalizeMinMax(dataset:: AbstractArray{<: Real, 2}, normalizationParameters:: NTuple{2, AbstractArray{<: Real, 2}})
    
    mins, maxs = normalizationParameters
    normalized_dataset = copy(dataset)
    normalized_dataset .= (normalized_dataset .- mins) ./ (maxs .- mins)
    normalized_dataset[:, vec(mins.==maxs)] .= 0;
    return normalized_dataset

end

function normalizeMinMax(dataset:: AbstractArray{<: Real, 2})
    
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    normalized_dataset = copy(dataset)
    return normalizeMinMax(normalized_dataset, normalizationParameters)
end

function normalizeZeroMean!(dataset:: AbstractArray{<: Real, 2}, normalizationParameters:: NTuple{2, AbstractArray{<: Real, 2}})
    
    meanv, stdv = normalizationParameters
    dataset .-= meanv
    dataset ./= stdv
    dataset[:, vec(stdv.==0)] .= 0;
    return dataset
end

function normalizeZeroMean!(dataset:: AbstractArray{<: Real, 2})
    
    normvalues = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean!(dataset, normvalues)
end

function normalizeZeroMean(dataset:: AbstractArray{<: Real, 2}, normalizationParameters:: NTuple{2, AbstractArray{<: Real, 2}})
    
    meanValues, stdValues = normalizationParameters
    normalized_dataset = copy(dataset)
    normalized_dataset .= (normalized_dataset .- meanValues) ./ stdValues
    normalized_dataset[:, vec(stdValues.==0)] .= 0;
    return normalized_dataset
end

function normalizeZeroMean(dataset:: AbstractArray{<: Real, 2})
    
    norm_values = calculateZeroMeanNormalizationParameters(dataset)
    normalized_dataset = copy(dataset)
    return normalizeZeroMean(normalized_dataset, norm_values)
end

# Funciones para clasificar outputs: probabilidades -> bools

function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    return reshape(outputs .>= threshold, :, 1)
end

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5) 

    dims = size(outputs)
    
    if dims[2] == 1
        return classifyOutputs(outputs[:]; threshold)
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2) # Obtener el índice de la clase con la probabilidad más alta
        outputs = falses(dims)
        outputs[indicesMaxEachInstance] .= true
        return outputs  
    end
end

# Funciones para calcular la accuracy de los outputs de un modelo.

# Primer caso: se pasa un vector booleano para target y otro para outputs

function accuracy(outputs:: AbstractArray{Bool, 1}, targets:: AbstractArray{Bool, 1})
    
    return mean(targets .== outputs)
end

# Segundo caso: se pasa una matriz de booleanos para targets y otra para outputs

function accuracy(outputs:: AbstractArray{Bool, 2}, targets:: AbstractArray{Bool, 2})

    dims = size(targets)

    if dims[2] == 1
        return accuracy(outputs[:], targets[:])

    elseif dims[2] > 2

        classComparison = targets .== outputs # Si la clase ha sido predicha correctamente, todos los elementos de la fila serán verdaderos (true == true, false == false, etc.)
        correctClassifications = all(classComparison, dims=2) 
        return mean(correctClassifications)

    end
end

# Tercer caso: se pasa un vector de targets booleanos y un vector de probabilidades como salida de la ANN.

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    
    return accuracy(outputs .>= threshold, targets)
end

# Cuarto caso: se pasa una matriz de targets booleanos y una matriz de probabilidades como salida de la ANN.

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5) 
           
    dims = size(targets)

    if dims[2] == 1
        return accuracy(outputs[:], targets[:]; threshold=threshold)

    elseif dims[2] > 2
        return accuracy(classifyOutputs(outputs; threshold=threshold), targets)
        
    end
end

# Función para construir una red neuronal

function buildClassANN(numInputs:: Int, topology:: AbstractArray{<:Int, 1}, numOutputs:: Int;
     transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    
    ann = Chain()
    numInputsLayer = numInputs
    
    # Construir capas ocultas usando las funciones de transferencia correspondientes
    for (numOutputsLayer, transferFunction) in zip(topology, transferFunctions)
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunction))
        numInputsLayer = numOutputsLayer
    end

    if numOutputs == 1 # problema de clasificacion binaria
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ));
    else # problema de clasificación multiclase
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs), softmax);
    end

    return ann
end

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