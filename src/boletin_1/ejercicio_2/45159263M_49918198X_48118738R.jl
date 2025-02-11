using DelimitedFiles
using Statistics
using Flux
using Flux.Losses

function oneHotEncoding(feature:: AbstractArray{<:Any, 1}, classes:: AbstractArray{<:Any, 1})

    num_classes = length(classes)

    if num_classes <= 2
        oneHot = reshape([feature.==classes[1]], :, 1)
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

function calculateMinMaxNortmalizationParameters(dataset:: AbstractArray{<:Real, 2})
    
    mins = minimum(dataset, dims=1)
    maxs = maximum(dataset, dims=1)
    return (mins, maxs)
end

function calculateZeroMeanNortmalizationParameters(dataset:: AbstractArray{<:Real, 2})
    
    meanValues = mean(dataset, dims=1)
    stdValues = std(dataset, dims=1)
    return (meanValues, stdValues)
end

function normalizeMinMax!(dataset:: AbstractArray{<: Real, 2}, normalizationParameters:: NTuple{2, AbstractArray{<: Real, 2}})
    
    mins, maxs = normalizationParameters
    dataset .= (dataset .- mins) ./ (maxs .- mins)
end

function normalizeMinMax!(dataset:: AbstractArray{<: Real, 2})
    
    normalizationParameters = calculateMinMaxNortmalizationParameters(dataset)
    normalizeMinMax!(dataset, normalizationParameters)
end

function normalizeMinMax(dataset:: AbstractArray{<: Real, 2}, normalizationParameters:: NTuple{2, AbstractArray{<: Real, 2}})
    
    mins, maxs = normalizationParameters
    normalized_dataset = copy(dataset)
    normalized_dataset .= (normalized_dataset .- mins) ./ (maxs .- mins)
    return normalized_dataset
end

function normalizeMinMax(dataset:: AbstractArray{<: Real, 2})
    
    normalizationParameters = calculateMinMaxNortmalizationParameters(dataset)
    normalized_dataset = copy(dataset)
    return normalizeMinMax(normalized_dataset, normalizationParameters)
end

function normalizeZeroMean!(dataset:: AbstractArray{<: Real, 2}, normalizationParameters:: NTuple{2, AbstractArray{<: Real, 2}})
    
    meanValues, stdValues = normalizationParameters
    dataset .= (dataset .- meanValues) ./ stdValues
    dataset[:, vec(stdValues.==0)] .= 0;
end

function normalizeZeroMean!(dataset:: AbstractArray{<: Real, 2})
    
    norm_values = calculateZeroMeanNortmalizationParameters(dataset)
    normalizeZeroMean!(dataset, norm_values)
end

function normalizeZeroMean(dataset:: AbstractArray{<: Real, 2}, normalizationParameters:: NTuple{2, AbstractArray{<: Real, 2}})
    
    meanValues, stdValues = normalizationParameters
    normalized_dataset = copy(dataset)
    normalized_dataset .= (normalized_dataset .- meanValues) ./ stdValues
    normalized_dataset[:, vec(stdValues.==0)] .= 0;
    return normalized_dataset
end

function normalizeZeroMean(dataset:: AbstractArray{<: Real, 2})
    
    norm_values = calculateZeroMeanNortmalizationParameters(dataset)
    normalized_dataset = copy(dataset)
    return normalizeZeroMean(normalized_dataset, norm_values)
end

function classifyOutputs(outputs:: AbstractArray{<: Real, 1}; threshold:: Real = 0.5)
    
    return outputs .>= threshold
end

function classifyOutputs(outputs:: AbstractArray{<: Real, 2}; threshold:: Real = 0.5)

    dims = size(outputs)
    
    if dims[2] == 1
        bool_vector = classifyOutputs(outputs[:1], threshold)
        return reshape(bool_vector, :, 1)
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2) # Obtener el índice de la clase con la probabilidad más alta
        outputs = falses(dims)
        outputs[indicesMaxEachInstance] .= true
        return outputs  
    end
end






# Primer caso: se pasa un vector booleano para target y otro para outputs

function accuracy(outputs:: AbstractArray{Bool, 1}, targets:: AbstractArray{Bool, 1})
    
    return mean(targets .== outputs)
end

# Segundo caso: se pasa una matriz de booleanos para targets y otra para outputs

function accuracy(outputs:: AbstractArray{Bool, 2}, targets:: AbstractArray{Bool, 2})

    dims = size(targets)

    if dims[2] == 1
        return accuracy(targets[:], outputs[:])

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
        return accuracy(targets[:1], outputs[:1])

    elseif dims[2] > 2
        return accuracy(classifyOutputs(outputs), targets)
        
    end
end











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

function trainClassANN(topology:: AbstractArray{<: Int, 1}, dataset:: Tuple{AbstractArray{<: Real, 2}, AbstractArray{Bool, 2}}, transferFunctions::AbstractArray{<:Function,1} = fill(σ, length(topology)), maxEpochs:: Int = 1000, minLoss:: Real = 0.0, learningRate:: Real = 0.01)

    inputs, targets = dataset
    inputs = Float32.(inputs')  # Convertir a Float32 y trasponer
    targets = (targets')   # Trasponer
    
    numInputs = size(inputs, 1)
    numOutputs = size(targets, 1)
    
    ann = buildClassANN(numInputs, topology, numOutputs, transferFunctions) # build ANN
    loss(model, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y) # loss function
    opt_state = Flux.setup(ADAM(learningRate), ann) # optimizer

    losses = Float32[] # losses array
    push!(losses, loss(ann, inputs, targets)) # append iteration 0 loss 
    
    for epoch in 1:maxEpochs
        
        Flux.train!(loss, ann, [(inputs, targets)], opt_state)
        current_loss = loss(ann, inputs, targets)
        push!(losses, current_loss)
        
        if current_loss ≤ minLoss
            break
        end
    end
    
    return ann, losses
end

function trainClassANN(topology:: AbstractArray{<: Int, 1}, dataset:: Tuple{AbstractArray{<: Real, 2}, AbstractArray{Bool, 1}}, transferFunctions::AbstractArray{<:Function,1} = fill(σ, length(topology)), maxEpochs:: Int = 1000, minLoss:: Real = 0.0, learningRate:: Real = 0.01)

    inputs, targets = dataset
    reshape!(targets, :, 1) 
    return trainClassANN(topology, (inputs, targets), transferFunctions, maxEpochs, minLoss, learningRate)
    
end