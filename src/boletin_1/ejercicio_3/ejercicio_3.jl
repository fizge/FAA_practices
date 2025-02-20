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