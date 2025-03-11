########################################################
##### ARCHIVO DE ENTREGAS PARA EJERCICIOS 4, 5 Y 6 #####
########################################################

# Ejercicio 4 -> métricas

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
    println(string(model))
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
