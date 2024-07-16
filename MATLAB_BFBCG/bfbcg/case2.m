format long g

mtxA = [
                    10.840188, 0.394383, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000;
                    0.394383, 10.783099, 0.798440, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000;
                    0.000000, 0.798440, 10.911648, 0.197551, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000;
                    0.000000, 0.000000, 0.197551, 10.335223, 0.768230, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000;
                    0.000000, 0.000000, 0.000000, 0.768230, 10.277775, 0.553970, 0.000000, 0.000000, 0.000000, 0.000000;
                    0.000000, 0.000000, 0.000000, 0.000000, 0.553970, 10.477397, 0.628871, 0.000000, 0.000000, 0.000000;
                    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.628871, 10.364784, 0.513401, 0.000000, 0.000000;
                    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.513401, 10.952229, 0.916195, 0.000000;
                    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.916195, 10.635712, 0.717297;
                    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.717297, 10.141603;
                ];


mtxBT = [-0.917206, 0.742276, 0.899125, -0.558456, 0.726547,0.343939, -0.319879, -0.901800, 0.898783,-0.885493;
                   -0.436044,-0.657422, -0.713946, -0.201146, -0.017785, -0.066716,-0.708516, 0.551868, 0.520759, -0.482016;
                   0.951020, -0.664174, 0.354639, 0.692534, 0.972838, 0.722408, -0.002730, -0.239378, 0.006314, -0.667044;
                   0.285601, 0.089107, -0.924769, 0.184726,0.530651, 0.801779, -0.471335, -0.789228, 0.899979,-0.572552;
                   -0.674721, -0.536065, -0.229974, -0.388667,0.262788, 0.752241, 0.544616, 0.554272, 0.304109, 0.065375];


mtxB = mtxBT';
mtxSolX = mtxB;

mtxI =  [1, 0, 0, 0, 0;
              0, 1, 0, 0, 0;
              0, 0, 1, 0, 0;
              0, 0, 0, 1, 0;
              0, 0, 0, 0, 1];

threshold = 1e-5;


% R <- B -AX
mtxR =  mtxB - mtxA * mtxSolX;
fprintf("\n\n~~mtxR~~\n\n");
disp(mtxR);

%Calculate original residual for the relative redsidual during the iteration
orgRsdl = calculateResidual(mtxR);
fprintf("\n\n~~Original residual: %f~~\n\n", orgRsdl);

% Z <- M * R
mtxZ = mtxR;
fprintf("\n\n~~mtxZ~~\n\n");
disp(mtxZ);

%P <- orth(Z)
mtxP = orth(mtxZ, threshold);
fprintf("\n\n~~mtxP~~\n\n");
disp(mtxP);

for wkr = 1 : 2
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
    mtxZ = mtxR;
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

