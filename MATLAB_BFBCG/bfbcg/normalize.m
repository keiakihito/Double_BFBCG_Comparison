function [mtxX_hat] = normalize(mtxX)
    % Initialize mtxX_hat with the same size as mtxX
    % size(mtxX) returns the row vector [numRows, numClms]
    mtxX_hat = zeros(size(mtxX));

    %normalize each column of matrix X
    for wkr = 1 : size(mtxX, 2)

        %Extract the ith column vector
        clmVec = mtxX(:, wkr);

        %compute the norm of the ith colunm vector
        sclarForNrm = 1 / norm(clmVec);

        % Normalize the ith column vector by deviding each elemnt by the norm
        clmVecNrm = sclarForNrm * clmVec;

        % Store the normalized column vector in the corresponding colum of mtxX_hat
        mtxX_hat(:, wkr) = clmVecNrm;
    end % end of for

end % end of function, normlize

