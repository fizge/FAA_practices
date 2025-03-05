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