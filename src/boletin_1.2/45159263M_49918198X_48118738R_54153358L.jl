########################################################
##### EJERCICIOS 2 Y 3 #################################
########################################################

using DelimitedFiles
using Statistics
using Flux
using Flux.Losses
using Random

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

function holdOut(N::Int, P::Real)
    @assert 0 <= P <= 1 "P debe estar entre 0 y 1"
    indices = randperm(N)  # Generate a random permutation of indexes
    split_point = floor(Int, N * (1 - P))  # Determines cut-off point
    return (indices[1:split_point], indices[split_point+1:end])  # Returns the tuple with the subdatasets
end


function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert 0 <= Pval + Ptest <= 1 "La suma de Pval y Ptest debe estar entre 0 y 1"
    (train_test, test) = holdOut(N, Ptest)  # Separates the test dataset
    Pval_adjusted = Pval / (1 - Ptest)  # Adjusts Pval for the remaining datasets 
    train, val = holdOut(length(train_test), Pval_adjusted)  # Separates validate dataset from training dataset
    return (train_test[train], train_test[val], test)  # Returns the three datasets
end


function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset:: Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
   (Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)),
   falses(0,size(trainingDataset[2],2))),
    testDataset:: Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
   (Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)),
   falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20)

    trainingInputs = Float32.(trainingDataset[1]') # Convertir inputs a float32
    validationInputs = Float32.(validationDataset[1]') # Convertir inputs a float32
    testInputs = Float32.(testDataset[1]') # Convertir inputs a float32

    trainingTargets = trainingDataset[2]' # Extraer labels
    validationTargets = validationDataset[2]' # Extraer labels
    testTargets = testDataset[2]' # Extraer labels
    
    numTrainingInputs = size(trainingInputs, 1)
    numTrainingOutputs = size(trainingTargets, 1)
    
    ann = buildClassANN(numTrainingInputs, topology, numTrainingOutputs; transferFunctions) # build ANN
    loss(model, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y) # loss function
    
    opt_state = Flux.setup(Adam(learningRate), ann) 

    trainingLosses = Float32[] # losses array
    validationLosses = Float32[] # validation losses
    testLosses = Float32[] # test losses array

    push!(trainingLosses, loss(ann, trainingInputs, trainingTargets)) # append epoch 0 loss 

    if !isempty(validationInputs)
        push!(validationLosses, loss(ann, validationInputs, validationTargets))
        bestValidationLoss = validationLosses[1]
        bestAnn = deepcopy(ann)
        epochsWithoutImprovement = 0
    end
    if !isempty(testInputs)
        push!(testLosses, loss(ann, testInputs, testTargets))
    end
    
    for epoch in 1:maxEpochs
      
        Flux.train!(loss, ann, [(trainingInputs, trainingTargets)], opt_state)
        currentTrainingLoss = loss(ann, trainingInputs, trainingTargets)
        push!(trainingLosses, currentTrainingLoss)
        
        if currentTrainingLoss <= minLoss
            break
        end

        # Case: conjunto de test
        if !isempty(testInputs)
            currentTestLoss = loss(ann, testInputs, testTargets)
            push!(testLosses, currentTestLoss)
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

            if epochsWithoutImprovement >= maxEpochsVal
                println("Parada temprana en epoch $epoch ")
                break
            end
        end
    end
    
    if !isempty(validationInputs)
        return (bestAnn, trainingLosses, validationLosses, testLosses)
    else
        return (ann, trainingLosses, validationLosses, testLosses)
    end
end;
   
   
function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset:: Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
   (Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)),
   falses(0)),
    testDataset:: Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
   (Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)),
   falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20) 

    newTrainingDataset = (trainingDataset[1], reshape(trainingDataset[2], :, 1))
    newValidationDataset = (validationDataset[1], reshape(validationDataset[2], :, 1))
    newTestDataset = (testDataset[1], reshape(testDataset[2], :, 1))

    return trainClassANN(topology, newTrainingDataset; validationDataset=newValidationDataset, testDataset=newTestDataset, 
                         transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, 
                         learningRate=learningRate, maxEpochsVal=maxEpochsVal)
end


########################################################
##### ARCHIVO DE ENTREGAS PARA EJERCICIOS 4, 5 Y 6 #####
########################################################

##################################
######## EJERCICIO 4.1 ###########
##################################

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

    true_positives = sum(outputs .& targets)
    false_positives = sum(outputs .& .!targets)
    true_negatives = sum(.!outputs .& .!targets)
    false_negatives = sum(.!outputs .& targets)

    accuracy = (true_positives + true_negatives) / length(outputs) # precision
    fail_rate = (false_positives + false_negatives) / length(outputs) # tasa de fallo
    recall = (true_positives == 0 && false_negatives == 0) ? 1 : true_positives / (true_positives + false_negatives) # sensibilidad
    especificity = (true_negatives == 0 && false_positives == 0) ? 1 : true_negatives / (true_negatives + false_positives) # especificidad
    precision = (true_positives == 0 && false_positives == 0) ? 1 : true_positives / (true_positives + false_positives) # valor predictivo positivo
    npv = (true_negatives == 0 && false_negatives == 0) ? 1 : true_negatives / (true_negatives + false_negatives) # valor predictivo negativo
    f1 = (recall == 0 && precision == 0) ? 0 : 2 * (precision * recall) / (precision + recall) # f1 score
    confussion_matrix = [true_negatives false_positives; false_negatives true_positives] # matriz de confusión

    return (accuracy, fail_rate, recall, especificity, precision, npv, f1, confussion_matrix)
end


function confusionMatrix(outputs::AbstractArray{<:Real,1},
    targets::AbstractArray{Bool,1}; threshold::Real=0.5)

    outputs = outputs .> threshold
    return confusionMatrix(outputs, targets)

end



##################################
######## EJERCICIO 4.2 ###########
##################################


function confusionMatrix(outputs::AbstractArray{Bool,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)

    num_classes = size(targets, 2)

    if size(outputs, 2) == size(targets, 2) && size(outputs, 2) > 2

        recall = zeros(num_classes) # Guardar memoria para sensibilidad
        especificity = zeros(num_classes) # Guardar memoria para especificidad
        precision = zeros(num_classes) # Guardar memoria para valor predictivo positivo
        npv = zeros(num_classes) # Guardar memoria para valor predictivo negativo
        f1 = zeros(num_classes) # Guardar memoria para f1 score

        for i in axes(outputs, 2)
            recall[i], especificity[i], precision[i], npv[i], f1[i], _ = confusionMatrix(outputs[:, i], targets[:, i])
        end

        confussion_matrix = [sum(outputs[:, i] .& targets[:, j]) for i in 1:num_classes, j in 1:num_classes] # Matriz de confusion

        if weighted
            w = vec(sum(targets, dims=1))/size(targets, 1) # Peso de cada clase
        else
            w = repeat([1/num_classes], num_classes) # Peso uniforme (= media aritmética)
        end

        # Calcular métricas según el peso de cada clase (weighted o media aritmética) 
        recall = sum(recall .* w)
        especificity = sum(especificity .* w)
        precision = sum(precision .* w)
        npv = sum(npv .* w)
        f1 = sum(f1 .* w)
        acc = accuracy(outputs, targets)
        fail_rate = 1 - acc

        return(acc, fail_rate, recall, especificity, precision, npv, f1, confussion_matrix)


    elseif size(outputs, 2) == size(targets, 2) && size(outputs, 2) == 1 # Clasificacion binaria
        return confusionMatrix(outputs[:], targets[:])
    end
end


function confusionMatrix(outputs::AbstractArray{<:Real,2},
    targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)

    outputs = classifyOutputs(outputs, threshold)
    return confusionMatrix(outputs, targets, weighted=weighted)

end
    

function confusionMatrix(outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1},
    classes::AbstractArray{<:Any,1}; weighted::Bool=true)

    @assert(all([in(label, classes) for label in vcat(targets, outputs)])) # Comprobar que las etiquetas son correctas
    @assert size(outputs, 1) == size(targets, 1)

    outputs = oneHotEncoding(outputs, classes)
    targets = oneHotEncoding(targets, classes)

    return confusionMatrix(outputs, targets, weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1}; weighted::Bool=true)

    classes = unique(vcat(outputs, targets))
    return confusionMatrix(outputs, targets, classes, weighted=weighted)
end


function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    cm = confusionMatrix(outputs, targets; weighted=weighted)
    println("Confusion Matrix:\n", cm[end])  # Último elemento es la matriz de confusión
    println("Accuracy: ", cm[1])
    println("Error Rate: ", cm[2])
    println("Sensitivity: ", cm[3])
    println("Specificity: ", cm[4])
    println("Positive Predictive Value (PPV): ", cm[5])
    println("Negative Predictive Value (NPV): ", cm[6])
    println("F1 Score: ", cm[7])
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    cm = confusionMatrix(outputs, targets; weighted=weighted)
    println("Confusion Matrix:\n", cm[end])  # Último elemento es la matriz de confusión
    println("Accuracy: ", cm[1])
    println("Error Rate: ", cm[2])
    println("Sensitivity: ", cm[3])
    println("Specificity: ", cm[4])
    println("Positive Predictive Value (PPV): ", cm[5])
    println("Negative Predictive Value (NPV): ", cm[6])
    println("F1 Score: ", cm[7])
end

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    cm = confusionMatrix(outputs, targets, classes; weighted=weighted)
    println("Confusion Matrix:\n", cm[end])  # Último elemento es la matriz de confusión
    println("Accuracy: ", cm[1])
    println("Error Rate: ", cm[2])
    println("Sensitivity: ", cm[3])
    println("Specificity: ", cm[4])
    println("Positive Predictive Value (PPV): ", cm[5])
    println("Negative Predictive Value (NPV): ", cm[6])
    println("F1 Score: ", cm[7])
end

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs))  # Extraer clases automáticamente
    printConfusionMatrix(outputs, targets, classes; weighted=weighted)
end


#import Pkg; Pkg.add("SymDoME")

using SymDoME

# Función para clasificación binaria (targets es un vector booleano)
function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
                        testInputs::AbstractArray{<:Real,2},
                        maximumNodes::Int)
    
    (trainingInputs, trainingTargets) = trainingDataset

    trainingInputs = Float64.(trainingInputs)
    testInputs = Float64.(testInputs)

    _, _, _, model = dome(trainingInputs, trainingTargets; maximumNodes = maximumNodes) 
    testOutputs = evaluateTree(model, testInputs)

    if isa(testOutputs, Real)
        testOutputs = repeat([testOutputs], size(testInputs, 1))
    end

    return testOutputs  
end

# Función para clasificación multiclase (targets es una matriz booleana)
function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
                        testInputs::AbstractArray{<:Real,2},
                        maximumNodes::Int)

    (trainingInputs, trainingTargets) = trainingDataset

    if size(trainingTargets, 2) == 1
        # Si hay una sola columna, se trata como clasificación binaria.
        testOutputs = trainClassDoME((trainingInputs, vec(trainingTargets)), testInputs, maximumNodes)
        return reshape(testOutputs, :, 1)

    elseif size(trainingTargets, 2) == 2
        # En caso de dos columnas, se utiliza una de ellas (por ejemplo, la primera)
        testOutputs = trainClassDoME((trainingInputs, vec(trainingTargets[:, 1])), testInputs, maximumNodes)
        return reshape(testOutputs, :, 1)
    
    else 
        # Estrategia "uno contra todos" para más de dos columnas.
        num_instances = size(testInputs, 1)
        num_classes = size(trainingTargets, 2)
        testOutputsMatrix = Array{Float64}(undef, num_instances, num_classes)

        for i in 1:num_classes
            testOutputs = trainClassDoME((trainingInputs, vec(trainingTargets[:, i])), testInputs, maximumNodes)
            testOutputsMatrix[:, i] = testOutputs
        end 
        return testOutputsMatrix
    end
end

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    testInputs::AbstractArray{<:Real,2},
    maximumNodes::Int)

    (trainingInputs, trainingTargets) = trainingDataset
    classes = unique(trainingTargets)

    testOutputs = Array{eltype(trainingTargets), 1}(undef, size(testInputs, 1))

    testOutputsDoME = trainClassDoME(
    (trainingInputs, oneHotEncoding(trainingTargets, classes)),
    testInputs, maximumNodes)

    testOutputsBool = classifyOutputs(testOutputsDoME; threshold=0)

    if length(classes) <= 2
        testOutputsBool = vec(testOutputsBool)
        testOutputs[testOutputsBool] .= classes[1]

        if length(classes) == 2
            testOutputs[.!testOutputsBool] .= classes[2]
        end
    elseif length(classes) > 2
        for numClass in axes(classes, 1)
            testOutputs[testOutputsBool[:, numClass]] .= classes[numClass]
        end
    end

    return testOutputs
    end



##########################
###### EJERCICIO 5 #######
##########################

function crossvalidation(N::Int64, k::Int64)

    folds = 1:k # number of folds
    k_folds = repeat(folds, Int(ceil(N/k))) # number of elements in each fold
    k_folds = k_folds[1:N] # remove the extra elements
    n_folds = shuffle!(k_folds) # shuffle the elements in each fold
    return n_folds

end

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)

    # Asegurarse que cada clase tiene al menos 10 representantes
    min_class_count = min(sum(targets), sum(.!targets))
    if min_class_count < 10     
       return
    end
     
    indices = collect(1:length(targets))
    indices[targets] = crossvalidation(sum(targets), k) # asignar a cada fila un valor de la lista de folds
    indices[.!targets] = crossvalidation(sum(.!targets), k) # Llamar a la funcion anterior con el numero de instancias negativas
    return indices
    
 end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)

    # Asegurarse que cada calse tiene al menos 10 representantes
    class_counts = vec(sum(targets, dims=1))  # Número de instancias por clase

    min_class_count = minimum(class_counts)  # Mínimo de instancias en cualquier clase
    if min_class_count < 10
        return
    end

    indices = collect(1:size(targets, 1))
    for i in axes(targets, 2)
        indices[targets[:,i]] = crossvalidation(sum(targets[:,i]), k) # asignar a cada fila un valor de la lista de folds
    end

    return indices
end


function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64) 

    return crossvalidation(oneHotEncoding(targets), k)

end

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    validationRatio::Real=0, maxEpochsVal::Int=20) 


    inputs, targets = dataset # Descomponer dataset
    classes = unique(targets) # Calcular las clases
    one_hot = oneHotEncoding(targets, classes) # OneHot de las clases

    folds = maximum(crossValidationIndices) # Calcular el número de folds


    accuracy = Float64[] # Vector de precisión (accuracy)
    fail_rate = Float64[] # Vector de tasa de error (error rate)
    recall = Float64[] # Vector de sensibilidad (recall)
    especificity = Float64[] # Vector de especificidad
    precision = Float64[] # Vector de VPP (precisión)
    npv = Float64[] # Vector de VPN
    f1 = Float64[] # Vector de F1
    confussion_matrix = zeros(length(classes), length(classes)) # Inicializar confussion matrix

    for fold in 1:folds

        # Extraer datos de entrenamiento y test según folds
        train_inputs = inputs[findall(crossValidationIndices .!= fold), :]
        train_targets = one_hot[findall(crossValidationIndices .!= fold), :]
        test_inputs = inputs[findall(crossValidationIndices .== fold), :]
        test_targets = one_hot[findall(crossValidationIndices .== fold), :]

        validation_inputs = Matrix{Float32}(undef, 0, size(train_inputs, 2)) # Inicializar validation inputs como matriz vacía
        validation_targets = Matrix{Bool}(undef, 0, size(train_targets, 2)) # Inicializar validation targets como matriz vacía

        if validationRatio > 0 # En caso de que tengamos validación

            v_ratio = validationRatio * (folds/(folds-1)) # Calcular ratio de validación adaptado.

            trainIndices, validationIndices = holdOut(size(train_inputs, 1), v_ratio) # Calcular indices de validación
            validation_inputs = train_inputs[validationIndices, :] # Extraer inputs de validación
            validation_targets = train_targets[validationIndices, :] # Extraer targets de validación
            train_inputs = train_inputs[trainIndices, :] # Extraer inputs de entrenamiento
            train_targets = train_targets[trainIndices, :] # Extraer targets de entrenamiento
        end

        # Crear nuevos vectores para las métricas de cada epoch de cada fold.
        
        acc_folf = []
        fail_rate_fold = []
        recall_fold = []
        especificity_fold = []
        precision_fold = []
        npv_fold = []
        f1_fold = []
        cnf_matrix_fold = Array{Float64}(undef, length(classes), length(classes), numExecutions)

        # Entrenar fold
        for i in 1:numExecutions

            # Entrenar ANN
            ann, _, _, _ = trainClassANN(
                topology, 
                (train_inputs, train_targets);
                validationDataset=(validation_inputs, validation_targets),
                testDataset=(test_inputs, test_targets),
                transferFunctions=transferFunctions, 
                maxEpochs=maxEpochs, 
                minLoss=minLoss, 
                learningRate=learningRate)

            # Calcular métricas
            metrics = confusionMatrix(ann(test_inputs')', test_targets)

            # Añadir métricas al registro.
            push!(acc_folf, metrics[1])
            push!(fail_rate_fold, metrics[2])
            push!(recall_fold, metrics[3])
            push!(especificity_fold, metrics[4])
            push!(precision_fold, metrics[5])
            push!(npv_fold, metrics[6])
            push!(f1_fold, metrics[7])
            cnf_matrix_fold[:, :, i] = metrics[8]

        end

        # Calcular métricas de cada fold
        push!(accuracy, mean(acc_folf))
        push!(fail_rate, mean(fail_rate_fold))
        push!(recall, mean(recall_fold))
        push!(especificity, mean(especificity_fold))
        push!(precision, mean(precision_fold))
        push!(npv, mean(npv_fold))
        push!(f1, mean(f1_fold))
        confussion_matrix += dropdims(mean(cnf_matrix_fold, dims=3), dims=3)
    end

    # Devolver media y desviación de las métricas
    return ((mean(accuracy), std(accuracy)), (mean(fail_rate), std(fail_rate)), (mean(recall), std(recall)), (mean(especificity), std(especificity)), (mean(precision), std(precision)), (mean(npv), std(npv)), (mean(f1), std(f1)), confussion_matrix)

end


##########################
###### EJERCICIO 6 #######
##########################


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict,
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1})

    # Normalizar claves: permitir tanto Strings como Symbols
    function get_param(dict, key)
        return get(dict, key, get(dict, Symbol(key), nothing))
    end

    if modelType == :ANN
        # Comprobar que existe el parámetro obligatorio
        topology = get_param(modelHyperparameters, "topology")
        @assert(topology !== nothing, "El parámetro topology' es obligatorio para ANN")
        @assert(isa(topology, AbstractArray{<:Int,1}), "topology debe ser un AbstractArray{<:Int,1}")

        # Crear un nuevo diccionario para los parámetros opcionales
        ann_params = Dict{Symbol, Any}()
        
        # Lista de parámetros opcionales con sus tipos esperados
        param_types = Dict(
            :numExecutions => Int,
            :transferFunctions => AbstractArray{<:Function,1},
            :maxEpochs => Int,
            :minLoss => Real,
            :learningRate => Real,
            :validationRatio => Real,
            :maxEpochsVal => Int
        )

        # Validar y agregar parámetros opcionales
        for (param, param_type) in param_types
            value = get_param(modelHyperparameters, param)
            if value !== nothing
                @assert isa(value, param_type) "$(param) debe ser de tipo $(param_type)"
                ann_params[param] = value
            end
        end

        # Llamar a ANNCrossValidation con los argumentos validados
        return ANNCrossValidation(
            topology, 
            dataset, 
            crossValidationIndices; 
            pairs(ann_params)...
        )

    else
        inputs, targets = dataset
        targets = string.(targets) # convertir a string
        classes = unique(targets)
        folds = maximum(crossValidationIndices)

        accuracy = Float64[]
        fail_rate = Float64[]
        recall = Float64[]
        especificity = Float64[]
        precision = Float64[]
        npv = Float64[]
        f1 = Float64[]
        confussion_matrix = zeros(length(classes), length(classes))

        for fold in 1:folds
            train_inputs = inputs[findall(crossValidationIndices .!= fold), :]
            train_targets = targets[findall(crossValidationIndices .!= fold), :]
            test_inputs = inputs[findall(crossValidationIndices .== fold), :]
            test_targets = targets[findall(crossValidationIndices .== fold), :]

            if modelType == :DoME
                maximumNodes = get_param(modelHyperparameters, "maximumNodes")
                @assert(maximumNodes !== nothing, "El parámetro 'maximumNodes' es obligatorio para DoME")
                @assert(isa(maximumNodes, Int), "maximumNodes debe ser un Int")

                model_output = trainClassDoME((train_inputs, train_targets[:]), test_inputs, maximumNodes)

            elseif modelType == :SVC

                svm_params = Dict{Symbol, Any}()

                C = get_param(modelHyperparameters, "C")
                @assert(C !== nothing, "El parámetro 'C' es obligatorio para SVM")
                @assert(isa(C, Real), "C debe ser un número Real")

                kernel = get_param(modelHyperparameters, "kernel")
                @assert(kernel !== nothing, "El parámetro 'kernel' es obligatorio para SVM")
                @assert(isa(kernel, String), "kernel debe ser un String")

                kernel_type = lowercase(kernel)
                kernel_map = Dict(
                    "linear" => LIBSVM.Kernel.Linear,
                    "rbf" => LIBSVM.Kernel.RadialBasis,
                    "radialbasis" => LIBSVM.Kernel.RadialBasis,
                    "sigmoid" => LIBSVM.Kernel.Sigmoid,
                    "poly" => LIBSVM.Kernel.Polynomial,
                    "polynomial" => LIBSVM.Kernel.Polynomial
                )

                @assert(haskey(kernel_map, kernel_type), "Kernel no soportado: $(kernel)")

                kernel = kernel_map[kernel_type]

                if kernel in [LIBSVM.Kernel.RadialBasis, LIBSVM.Kernel.Sigmoid, LIBSVM.Kernel.Polynomial]
                    gamma = get_param(modelHyperparameters, "gamma")
                    @assert(gamma !== nothing, "El parámetro 'gamma' es obligatorio para $(kernel_type) SVM")
                    @assert(isa(gamma, Real), "gamma debe ser un número Real")
                    svm_params[:gamma] = Float64(gamma)
                end

                if kernel in [LIBSVM.Kernel.Sigmoid, LIBSVM.Kernel.Polynomial]
                    coef0 = get_param(modelHyperparameters, "coef0")
                    @assert(coef0 !== nothing, "El parámetro 'coef0' es obligatorio para $(kernel_type) SVM")
                    @assert(isa(coef0, Real), "coef0 debe ser un número Real")
                    svm_params[:coef0] = Float64(coef0)
                end

                if kernel == LIBSVM.Kernel.Polynomial
                    degree = get_param(modelHyperparameters, "degree")
                    @assert(degree !== nothing, "El parámetro 'degree' es obligatorio para Polynomial SVM")
                    @assert(isa(degree, Int), "degree debe ser un Int")
                    svm_params[:degree] = Int32(degree)
                end

                model = SVMClassifier(kernel=kernel, cost=Float64(C); svm_params...)
                mach = machine(model, MLJ.table(train_inputs), categorical(train_targets[:]))
                MLJ.fit!(mach, verbosity=0)
                model_output = MLJ.predict(mach, MLJ.table(test_inputs))

            elseif modelType == :DecisionTreeClassifier

                max_depth = get_param(modelHyperparameters, "max_depth")
                rng = get_param(modelHyperparameters, "rng")
                @assert(max_depth !== nothing && rng !== nothing, "max_depth y rng son obligatorios para DecisionTree")
                
                model = DTClassifier(max_depth=max_depth, rng=rng)
                mach = machine(model, MLJ.table(train_inputs), categorical(train_targets[:]))
                MLJ.fit!(mach, verbosity=0)
                output = MLJ.predict(mach, MLJ.table(test_inputs))
                model_output = mode.(output)

            elseif modelType == :KNNClassifier
                
                k = get_param(modelHyperparameters, "k")
                @assert(k !== nothing, "El parámetro 'k' es obligatorio para KNN")
                
                model = kNNClassifier(K = k)
                mach = machine(model, MLJ.table(train_inputs), categorical(train_targets[:]))
                MLJ.fit!(mach, verbosity=0)
                output = MLJ.predict(mach, MLJ.table(test_inputs))
                model_output = mode.(output)
            end

            # Calcular métricas del fold
            metrics = confusionMatrix(model_output, test_targets[:], classes)
            push!(accuracy, metrics[1])
            push!(fail_rate, metrics[2])
            push!(recall, metrics[3])
            push!(especificity, metrics[4])
            push!(precision, metrics[5])
            push!(npv, metrics[6])
            push!(f1, metrics[7])
            confussion_matrix += metrics[8]
        end

        return (
            (mean(accuracy), std(accuracy)),
            (mean(fail_rate), std(fail_rate)),
            (mean(recall), std(recall)),
            (mean(especificity), std(especificity)),
            (mean(precision), std(precision)),
            (mean(npv), std(npv)),
            (mean(f1), std(f1)),
            confussion_matrix
           )
    end
end