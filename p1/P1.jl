using DelimitedFiles
using Statistics
dataset = readdlm("iris.data",',');

inputs = dataset[:,1:4];
inputs = convert(Array{Float32,2},inputs);

targets = dataset[:,5];
clases = unique(targets);
numclases = length(clases);
println("Targets ANTES de binarizar: ", size(targets,1), "x", size(targets,2), " de tipo ", typeof(targets));

if numclases<=2
    targets = reshape(targets.==clases[1], :, 1);

else
    oneHot = convert(BitArray{2}, hcat([instance.==clases for instance in targets]...)');
    targets = oneHot;
end;
println("Targets después de binarizar: ", size(targets,1), "x", size(targets,2), " de tipo ", typeof(targets));

@assert (size(inputs,1)==size(targets,1)) "ERROR NUM FILAS ENTRE ENTRADA Y SALIDA"


# NORMALIZACIÓN MAX-MIN
normalizationParameters = (minimum(inputs, dims=1), maximum(inputs, dims=1));

minValues = normalizationParameters[1];
maxValues = normalizationParameters[2];

inputs .-= minValues;
inputs ./= (maxValues .- minValues);
inputs[:, vec(minValues.==maxValues)] .= 0;

# NORMALIZACIÓN MEDIA 0 DV TÍPICA
normalizationParameters = ( mean(inputs, dims=1), std(inputs, dims=1));
avgValues = normalizationParameters[1];
stdValues = normalizationParameters[2];
inputs .-= avgValues;
inputs ./= stdValues;
inputs[:, vec(stdValues.==0)] .= 0;