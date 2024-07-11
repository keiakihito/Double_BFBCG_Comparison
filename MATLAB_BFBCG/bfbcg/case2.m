format long g

mtxA = single([ 10.840188, 0.394383, 0.000000, 0.000000, 0.000000,
                            0.394383, 10.783099, 0.798440, 0.000000, 0.000000,
                            0.000000, 0.798440, 10.911648, 0.197551, 0.000000,
                            0.000000, 0.000000, 0.197551, 10.335223, 0.768230,
                            0.000000, 0.000000, 0.000000, 0.768230, 10.277775 ]);

mtxSolXT = single([0, 0, 0, 0, 0;
                                0, 0, 0, 0, 0;
                                0, 0, 0, 0, 0]);
mtxSolX = mtxSolXT';

mtxBT = single([1.1, 1.2, 1.3, 1.4, 1.5;
                           1.6, 1.7, 1.8, 1.9, 2.0;
                           2.1, 2.2, 2.3, 2.4, 2.5]);

mtxB = mtxBT';

mtxI =  single([1, 0, 0, 0, 0;
                          0, 1, 0, 0, 0;
                          0, 0, 1, 0, 0;
                          0, 0, 0, 1, 0;
                          0, 0, 0, 0, 1]);
threshold = 1e-5;


% R <- B -AX
mtxR =  mtxB - mtxA * mtxSolX;
fprintf("\n\n~~mtxR~~\n\n");
disp(mtxR);

%Calculate original residual for the relative redsidual during the iteration
orgRsdl = calculateResidual(mtxR);
fprintf("\n\n~~Original residual: %f~~\n\n", orgRsdl);

% Z <- M * R
mtxZ = mtxI * mtxR;
fprintf("\n\n~~mtxZ~~\n\n");
disp(mtxZ);

%P <- orth(Z)
mtxP = orth(mtxZ, threshold);
fprintf("\n\n~~mtxP~~\n\n");
disp(mtxP);

for wkr = 1 : 5
    %Q <- AP
    mtxQ = mtxA * mtxP;
    fprintf("\n\n~~mtxQ~~\n\n");
    disp(mtxQ);

    %Set up (P'Q)^{-1}, (P'R)
    mtxPTQ_Inv = inv(mtxP' * mtxQ);
    mtxPTR = mtxP' * mtxR;
    fprintf("\n\n~~mtxPQ_Inv~~\n\n");
    disp(mtxPTQ_Inv);
    fprintf("\n\n~~mtxP'R~~\n\n");
    disp(mtxPTR);

    %Aplha <- (P'Q)^{-1} * (P'R)
    mtxAlph = mtxPTQ_Inv * mtxPTR;
    fprintf("\n\n~~mtxAlph~~\n\n");
    disp(mtxAlph);

    %X_{i+1} <- x_{i} + P * alpha
    mtxSolX = mtxSolX + (mtxP * mtxAlph);
    fprintf("\n\n~~mtxSolX~~\n\n");
    disp(mtxSolX);

    %R_{i+1} <- R_{i} - Q * alpha
    mtxR = mtxR - (mtxQ * mtxAlph);
    fprintf("\n\n~~mtxR~~\n\n");
    disp(mtxR);

    % Calculate relative residue
    crrntRsdl = calculateResidual(mtxR);
    rltvRsdl = crrntRsdl / orgRsdl;
    fprintf("\n\n~~relative residue: %f~~~ \n\n", rltvRsdl);

    %If converged within tol, then stop

    %Z <- MR
    mtxZ = mtxI * mtxR;
    fprintf("\n\n~~mtxZ~~\n\n");
    disp(mtxZ);

    %(Q'Z)
    mtxQTZ = (mtxQ' * mtxZ);
    fprintf("\n\n~~mtxQTZ~~\n\n");
    disp(mtxQTZ);

    %beta <- -(P'Q)^{-1} * (Q'Z)
    mtxBta = -(mtxPTQ_Inv) * mtxQTZ;
    fprintf("\n\n~~mtxBta~~\n\n");
    disp(mtxBta);

    %mtxZ + P * beta
    mtxZ_pls_PBta = mtxZ + mtxP * mtxBta;
    fprintf("\n\n~~mtxZ + p * beta~~\n\n");
    disp(mtxZ_pls_PBta);

    %P_{i+1} <- orth(Z + P*beta)
    mtxP = orth((mtxZ + mtxP * mtxBta), threshold);
    fprintf("\n\n~~mtxP~~\n\n");
    disp(mtxP);
end

fprintf("\n\n~~After iteration~~\n");
fprintf("\n~~mtxX_Sol~~\n\n");
disp(mtxSolX);

