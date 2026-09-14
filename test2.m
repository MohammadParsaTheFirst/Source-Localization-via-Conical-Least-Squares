clear; clc; close all;

%% ===================== PARAMETERS =====================
M      = 30;                       % number of Monte-Carlo runs
N_list = [15, 25, 35, 45];%, 55];     % different trajectory lengths
rho    = 1e-6;

x_0      = [1; 2; 3];
x_1      = [100; 200; 300];
x_source = [-45; 567; -456];

%% ===================== STORAGE ========================
% Results will be stored as:
%   Error(m,j)   - localization error of run m, length N_list(j)
%   Rank(m,j)
%   Loss(m,j)
%   Cond(m,j)
%   Eigs{m,j}    - eigenvalues of the 3x3 matrix
numN   = length(N_list);
Error  = zeros(M, numN);
Rank   = zeros(M, numN);
LossV  = zeros(M, numN);
Cond   = zeros(M, numN);
Eigs   = cell(M, numN);

%% ===================== MAIN LOOPS =====================
for m = 1:M
    fprintf('=== Monte-Carlo run %d / %d ===\n', m, M);
    
    for j = 1:numN
        n = N_list(j);
        
        % Generate a new random trajectory each time
        [x_traj, x_heading] = traj_gen(n, x_0, x_1, 0.15, 0.85, 0.15, 0.85);
        x_angles = calc_angles(x_traj, x_source, x_heading);
        
        % Solve the SDP ---> use either sdp1 or sdp2 as the solver of the optim problem  
        [u_opt, Z, mat] = sdp2(x_traj, x_heading, x_angles, n, rho); % ------------------------------------------------
        
        % ----- record important quantities -----
        Error(m,j)  = norm(u_opt - x_source);
        Rank(m,j)   = rank(Z, 1e-6);          % tolerance for numerical rank
        LossV(m,j)  = compute_loss(x_traj, x_heading, x_angles, Z, rho);
        Cond(m,j)   = cond(mat);
        Eigs{m,j}   = eig(mat);
        
        % Optional: print progress
        fprintf('  N = %2d | Err = %.3e | Rank = %d | Loss = %.3e | Cond = %.2e\n', ...
                n, Error(m,j), Rank(m,j), LossV(m,j), Cond(m,j));
    end
end

%% ===================== VISUALIZATION ==================
% 1. Localization error vs N (mean ± std)
figure('Name','Localization Error','Position',[100 100 800 500]);
meanErr = mean(Error,1);
stdErr  = std(Error,0,1);
errorbar(N_list, meanErr, stdErr, '-o', 'LineWidth',1.8, 'MarkerSize',8);
grid on; xlabel('Number of samples N'); ylabel('||u_{opt} - x_{source}||');
title(sprintf('Localization Error over %d Monte-Carlo runs', M));
set(gca,'YScale','log');          % usually better for error plots

% 2. Rank statistics
figure('Name','Rank of Z','Position',[150 150 800 500]);
boxplot(Rank, 'Labels', string(N_list));
xlabel('Number of samples N'); ylabel('rank(Z)');
title('Distribution of rank(Z)');
grid on;

% 3. Loss values
figure('Name','Objective / Loss','Position',[200 200 800 500]);
meanLoss = mean(LossV,1);
stdLoss  = std(LossV,0,1);
errorbar(N_list, meanLoss, stdLoss, '-s', 'LineWidth',1.8, 'MarkerSize',8);
grid on; xlabel('Number of samples N'); ylabel('Loss = tr(CZ) + \rho tr(Z)');
title('Objective value');

% 4. Condition number of the 3×3 matrix
figure('Name','Condition number','Position',[250 250 800 500]);
meanCond = mean(Cond,1);
stdCond  = std(Cond,0,1);
errorbar(N_list, meanCond, stdCond, '-d', 'LineWidth',1.8, 'MarkerSize',8);
grid on; xlabel('Number of samples N'); ylabel('cond(mat)');
title('Condition number of Z_{uu}-u u^T');
set(gca,'YScale','log');

% 5. (Optional) Eigenvalues of the residual matrix – one figure per N
figure('Name','Eigenvalues of residual matrix','Position',[300 100 1000 600]);
for j = 1:numN
    subplot(2, ceil(numN/2), j);
    allEigs = cell2mat(Eigs(:,j)');   % 3 × M
    plot(1:M, allEigs(1,:), 'r.', 1:M, allEigs(2,:), 'g.', 1:M, allEigs(3,:), 'b.');
    title(sprintf('N = %d', N_list(j)));
    xlabel('run'); ylabel('eigenvalue');
    grid on; legend('\lambda_1','\lambda_2','\lambda_3','Location','best');
end
sgtitle('Eigenvalues of mat = Z_{uu}-uu^T across Monte-Carlo runs');

%% ===================== HELPER FUNCTIONS ===============
function angles = calc_angles(x_traj, x_sensor, x_heading)
    d = x_sensor - x_traj;
    d_norm = d ./ sqrt(sum(d.^2,1));
    cos_theta = sum(d_norm .* x_heading, 1);
    angles = acos(max(-1, min(1, cos_theta)));
end

function L = compute_loss(x_traj, x_heading, x_angles, Z, rho)
    n = size(x_traj,2);
    idx_u = 1:3;
    idx_r = 4:3+n;
    idx_1 = n+4;
    
    C = zeros(idx_1);
    for i = 1:n
        c_i = zeros(idx_1,1);
        c_i(idx_u)   = -sin(x_angles(i)) * x_heading(:,i);
        c_i(idx_r(i)) =  cos(x_angles(i));
        c_i(idx_1)   =  sin(x_angles(i)) * (x_heading(:,i)' * x_traj(:,i));
        C = C + c_i * c_i';
    end
    L = trace(C * Z) + rho * trace(Z);   % same objective used in CVX
    L = L/n;
end

function [x, T] = traj_gen(n, p0, p3, alpha1, alpha2, K1, K2)
    t  = linspace(0,1,n);
    v1 = randn(3,1);  v2 = randn(3,1);
    k1 = unifrnd(K1,K2);  k2 = unifrnd(K1,K2);
    
    d  = p3 - p0;  L = norm(d);  u = d/L;
    n1 = v1 - (v1'*u)*u;  n1 = n1/norm(n1);
    n2 = v2 - (v2'*u)*u;  n2 = n2/norm(n2);
    
    p1 = p0 + alpha1*d + k1*L*n1;
    p2 = p0 + (1-alpha2)*d + k2*L*n2;
    
    x = (1-t).^3 .* p0 + 3*(1-t).^2.*t .* p1 + ...
        3*(1-t).*t.^2 .* p2 + t.^3 .* p3;
    
    dx = 3*(1-t).^2.*(p1-p0) + 6*(1-t).*t.*(p2-p1) + 3*t.^2.*(p3-p2);
    speed = sqrt(sum(dx.^2,1));
    T = dx ./ speed;
end

function [u_opt, Z, mat] = sdp(x_traj, x_heading, x_angles, N, rho)
    n = N;
    idx_u = 1:3;
    idx_r = 4:3+n;
    idx_1 = n+4;
    
    C = zeros(idx_1);
    for i = 1:n
        c_i = zeros(idx_1,1);
        c_i(idx_u)    = -sin(x_angles(i)) * x_heading(:,i);
        c_i(idx_r(i)) =  cos(x_angles(i));
        c_i(idx_1)    =  sin(x_angles(i)) * (x_heading(:,i)' * x_traj(:,i));
        C = C + c_i * c_i';
    end
    
    cvx_begin sdp quiet
        variable Z(idx_1,idx_1) symmetric
        minimize( trace(C*Z) + rho*trace(Z) )
        subject to
            Z == semidefinite(idx_1);
            Z(idx_1,idx_1) == 1;
            for i = 1:n
                P_i = eye(3) - x_heading(:,i)*x_heading(:,i)';
                Z(idx_r(i),idx_r(i)) == ...
                    trace(P_i*Z(idx_u,idx_u)) ...
                    - 2*x_traj(:,i)'*P_i*Z(idx_u,idx_1) ...
                    + x_traj(:,i)'*P_i*x_traj(:,i);
                Z(idx_r(i),idx_1) >= 0;
            end
    cvx_end
    
    u_opt = Z(idx_u, idx_1);
    mat   = Z(idx_u,idx_u) - u_opt*u_opt';
end


function [u_opt, Z, mat] = sdp2(x_traj, x_heading, x_angles, N, rho)
    n = N;
    idx_u = 1:3;
    idx_r = 4:3+n;
    idx_1 = n+4;
    dim  = idx_1;
    
    % Build the constant matrix C
    C = zeros(dim);
    for i = 1:n
        c_i = zeros(dim,1);
        c_i(idx_u)    = -sin(x_angles(i)) * x_heading(:,i);
        c_i(idx_r(i)) =  cos(x_angles(i));
        c_i(idx_1)    =  sin(x_angles(i)) * (x_heading(:,i)' * x_traj(:,i));
        C = C + c_i * c_i';
    end
    
    % ---------- initial SDP solve for Z ----------
    cvx_begin sdp quiet
    variable Z(dim,dim) symmetric
    minimize( trace(C*Z) + rho*trace(Z) )
    subject to
    Z == semidefinite(dim);
    Z(idx_1,idx_1) == 1;
    for i = 1:n
        P_i = eye(3) - x_heading(:,i)*x_heading(:,i)';
        Z(idx_r(i),idx_r(i)) == ...
            trace(P_i*Z(idx_u,idx_u)) ...
            - 2*x_traj(:,i)'*P_i*Z(idx_u,idx_1) ...
            + x_traj(:,i)'*P_i*x_traj(:,i);
        Z(idx_r(i),idx_1) >= 0;
    end
    cvx_end
    
    % ---------- coordinate / alternating descent: Z <-> v ----------
    max_iter = 15;          % number of outer alternations
    v = Z(:,idx_1);         % initial vector (last column of Z)
    v = v / norm(v);        % normalise so that v(end) ≈ 1
    
    for iter = 1:max_iter
        % ---- (1) fix v, re-optimise Z (still SDP but warm-started) ----
        cvx_begin sdp quiet
        variable Z(dim,dim) symmetric
        minimize( trace(C*Z) + rho*trace(Z) )
        subject to
        Z == semidefinite(dim);
        Z(idx_1,idx_1) == 1;
        % soft rank-1 encouragement: force Z close to v*v'
        % (can be strengthened by adding ||Z - v*v'||_* or a penalty)
        for i = 1:n
            P_i = eye(3) - x_heading(:,i)*x_heading(:,i)';
            Z(idx_r(i),idx_r(i)) == ...
                trace(P_i*Z(idx_u,idx_u)) ...
                - 2*x_traj(:,i)'*P_i*Z(idx_u,idx_1) ...
                + x_traj(:,i)'*P_i*x_traj(:,i);
            Z(idx_r(i),idx_1) >= 0;
        end
        cvx_end
    
        % ---- (2) fix Z, update v by principal eigenvector of Z ----
        [V,D] = eig(Z);
        [~,idx] = max(abs(diag(D)));
        v = V(:,idx);
        % enforce the homogeneous coordinate ≈ +1
        if v(idx_1) < 0
            v = -v;
        end
        v = v / v(idx_1);   % scale so that last entry = 1
    end
    
    % final extraction
    u_opt = v(idx_u);               % or Z(idx_u,idx_1)
    mat   = Z(idx_u,idx_u) - u_opt*u_opt';
end
