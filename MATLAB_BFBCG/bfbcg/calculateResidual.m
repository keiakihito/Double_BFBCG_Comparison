function originalResidual = calculateResidual(mtxR)
    % Extract the first column vector from mtxR
    firstColumn = mtxR(:, 1);

    % Calculate the dot product
    dotProduct = firstColumn' * firstColumn;

    % Calculate the square root
    originalResidual = sqrt(dotProduct);
end
