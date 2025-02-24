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
    return outputs .>= threshold
end

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5) 

    dims = size(outputs)
    
    if dims[2] == 1
        return reshape(classifyOutputs(outputs[:]; threshold), : ,1)
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

# Funciones de holdout para dividir el dataset

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

# Funciones de entrenamiento con early stopping

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    
    trainingInputs = Float32.(trainingDataset[1]')
    validationInputs = Float32.(validationDataset[1]')
    testInputs = Float32.(testDataset[1]')

    trainingTargets = trainingDataset[2]'
    validationTargets = validationDataset[2]'
    testTargets = testDataset[2]'
    
    numTrainingInputs = size(trainingInputs, 1)
    numTrainingOutputs = size(trainingTargets, 1)
    
    ann = buildClassANN(numTrainingInputs, topology, numTrainingOutputs; transferFunctions) # build ANN
    loss(model, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y) # loss function
    opt_state = Flux.setup(ADAM(learningRate), ann) # optimizer

    trainingLosses = Float32[] # losses array
    validationLosses = Float32[]
    testLosses = Float32[]

    push!(trainingLosses, loss(ann, trainingInputs, trainingTargets)) # append iteration 0 loss 
    push!(validationLosses, loss(ann, validationInputs, validationTargets))
    push!(testLosses, loss(ann, testInputs, testTargets))

    bestValidationLoss = Inf
    bestAnn = deepcopy(ann)
    epochsWithoutImprovement = 0

    for epoch in 1:maxEpochs
        # ENTRENAMIENTO
        Flux.train!(loss, ann, [(trainingInputs, trainingTargets)], opt_state)
        currentTrainingLoss = loss(ann, trainingInputs, trainingTargets)
        push!(trainingLosses, currentTrainingLoss)
        
        if currentTrainingLoss <= minLoss
            break
        end
    
        # VALIDACIÓN (si hay conjunto de validación)
        if !isempty(validationInputs)
            currentValidationLoss = loss(ann, validationInputs, validationTargets)
            push!(validationLosses, currentValidationLoss)
            
            # PARADA TEMPRANA: Si el loss en validación mejora, actualizar mejor modelo
            if currentValidationLoss < bestValidationLoss
                bestValidationLoss = currentValidationLoss
                bestAnn = deepcopy(ann)
                epochsWithoutImprovement = 0  # Reiniciar el contador de epochs sin mejora
            else
                epochsWithoutImprovement += 1
            end
        end
    
        # PRUEBA (si hay conjunto de test)
        if !isempty(testInputs)
            currentTestLoss = loss(ann, testInputs, testTargets)
            push!(testLosses, currentTestLoss)
        end
    
        # CRITERIO DE PARADA TEMPRANA
        if epochsWithoutImprovement >= maxEpochsVal
            println("Parada temprana en epoch $epoch")
            break
        end
    end
    if !isempty(validationInputs)
        return bestAnn, trainingLosses, validationLosses, testLosses
    else
        return ann, trainingLosses, validationLosses, testLosses
    end
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    newTrainingDataset = (trainingDataset[1], reshape(trainingDataset[2], :, 1))
    newValidationDataset = (validationDataset[1], reshape(validationDataset[2], :, 1))
    newTestDataset = (testDataset[1], reshape(testDataset[2], :, 1))

    return trainClassANN(topology, newTrainingDataset; validationDataset=newValidationDataset, testDataset=newTestDataset, 
                         transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, 
                         learningRate=learningRate, maxEpochsVal=maxEpochsVal)
end;