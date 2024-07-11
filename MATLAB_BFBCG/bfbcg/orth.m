function [mtxY_Hat] = orth(mtxZ, threshold)


%    fprintf("\n\n~~mtxZ~~\n");
%    disp(mtxZ);
%
%    fprintf("\n\n~~Transpose mtxZ~~\n");
%    disp(mtxZ');

    % Make square matrix with matrix Z' * matrix Z to
    mtxS = mtxZ' * mtxZ;
%    fprintf("\n\n~~mtxS~~\n");
%    disp(mtxS);

    % Perform SVD
    [mtxU, sngVals, mtxV] = svd(mtxS, 'econ');
%    fprintf("\n\n~~mtxU~~\n");
%    disp(mtxU);

%    fprintf("\n\n~~sngVals~~\n");
%    disp(sngVals);
%
%    fprintf("\n\n~~mtxVT~~\n");
%    disp(mtxV');
%
%    fprintf("\n\n~~mtxV~~\n");
%    disp(mtxV);

    % Count significant values and set rank, r
    r = diag(sngVals) > threshold;
    rankSum = sum(r);
%    fprintf("\n\nCurrent Rank = ");
%    disp(rankSum);

    % Truncate matrix V matrices with extracted rank
    mtxV_Trnc = mtxV(:, r);
%    fprintf("\n\n~~mtxV Truncated~~\n");
%    disp(mtxV_Trnc);

    % Obtain orthonormal set with rank r Y <- Z * V_
    mtxY = mtxZ * mtxV_Trnc;
%    fprintf("\n\n~~mtxY~~\n");
%    disp(mtxY);

    % Normalize mtxY
    mtxY_Hat = normalize(mtxY);
%    fprintf("\n\n~~mtxY Hat <- orth(*)~~\n");
%    disp(mtxY_Hat);

%    % Orthogonality check
%    orth_check = mtxY_Hat' * mtxY_Hat;
%    fprintf("\n\n~~Orthogonality Check (should be close to identity matrix)~~\n");
%    disp(orth_check);
end
