## Reparto de Funciones para 4 Personas

### **Persona 1**
    confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
### **Persona 2**
    confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
### **Persona 3**
    trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
### **Persona 4**
    trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)