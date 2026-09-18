clear, clc;
x_0 = [1; 2; 3];
x_1 = [100; 200; 300];
%x_source = [45; -822; 322];
%x_source = [-45; 567; -456];
%x_source = [1200; -567; -900];
x_source = [-5000; -2000; 4000];

N = [15, 25, 35, 45, 55]; %, 50, 75, 100, 250];
Errors_sdp  = [];
Errors_sdp4 = [];
Ranks_sdp   = [];
Ranks_sdp4  = [];

% Noise standard deviation (radians). Adjust as needed.
sigma_noise = 0.04;   % .02 rad ~1.15 degrees

figure;
hold on; grid on;view(3);
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Trajectory + True Source + Localized Estimates');

for j = 1:length(N)
    n   = N(j);
    rho = 10^(-4.5);


    % Generate trajectory and headings
    [x_traj, x_heading] = traj_gen(n, x_0, x_1, 0.15, 0.85, 0.15, 0.85);

    % Plot trajectory (only once is enough, but harmless)
    plot3(x_traj(1,:), x_traj(2,:), x_traj(3,:), 'b-', 'LineWidth', 1.5);

    % True source
    plot3(x_source(1), x_source(2), x_source(3), 'ro', ...
          'MarkerSize', 10, 'LineWidth', 2);

    % Perfect angles
    x_angles_true = calc_angles(x_traj, x_source, x_heading);

    % ---------- Add Gaussian noise ----------
    x_angles = x_angles_true + sigma_noise * randn(1, n);
    % Keep angles inside [0, pi] (acos range)
    x_angles = max(0, min(pi, x_angles));

    % ===================== Classic SDP =====================
    [u_opt_sdp, Z_sdp, mat_sdp] = sdp(x_traj, x_heading, x_angles, n, rho);
    plot3(u_opt_sdp(1), u_opt_sdp(2), u_opt_sdp(3), 'gs', ...
          'MarkerSize', 8, 'LineWidth', 1.5);
    text(u_opt_sdp(1), u_opt_sdp(2), u_opt_sdp(3), ...
         ['S' num2str(j)], 'FontSize', 8, 'Color', 'g');

    Errors_sdp(end+1) = norm(u_opt_sdp - x_source);
    Ranks_sdp(end+1)  = rank(Z_sdp, 1e-6);

    % ===================== SDP4 (penalty-CCP) =====================
    [u_opt_sdp4, Z_sdp4, mat_sdp4] = sdp4(x_traj, x_heading, x_angles, n, rho);
    plot3(u_opt_sdp4(1), u_opt_sdp4(2), u_opt_sdp4(3), 'm^', ...
          'MarkerSize', 8, 'LineWidth', 1.5);
    text(u_opt_sdp4(1), u_opt_sdp4(2), u_opt_sdp4(3), ...
         ['4' num2str(j)], 'FontSize', 8, 'Color', 'm');

    Errors_sdp4(end+1) = norm(u_opt_sdp4 - x_source);
    Ranks_sdp4(end+1)  = rank(Z_sdp4, 1e-6);

    % Optional diagnostics
    fprintf('----- N = %d -----\n', n);
    fprintf('SDP   error = %.4f   rank(Z) = %d\n', Errors_sdp(end),  Ranks_sdp(end));
    fprintf('SDP4  error = %.4f   rank(Z) = %d\n', Errors_sdp4(end), Ranks_sdp4(end));
    fprintf('cond(mat_sdp)  = %.2e\n', cond(mat_sdp));
    fprintf('cond(mat_sdp4) = %.2e\n', cond(mat_sdp4));
    disp('------------------');
end

legend({'Trajectory', 'True source', 'SDP', 'SDP4'}, 'Location', 'best');

disp('Errors SDP :');  disp(Errors_sdp);
disp('Errors SDP4:');  disp(Errors_sdp4);
disp('Ranks  SDP :');  disp(Ranks_sdp);
disp('Ranks  SDP4:');  disp(Ranks_sdp4);

%% ------------------------------------------------------------------
function [angles] = calc_angles(x_traj, x_sensor, x_heading)
    d_norm   = (x_sensor - x_traj) ./ sqrt(sum((x_sensor - x_traj).^2, 1));
    cos_theta = sum(d_norm .* x_heading, 1);
    angles   = acos(max(-1, min(1, cos_theta)));
end

function [x, T] = traj_gen(n, p0, p3, alpha1, alpha2, K1, K2)
    t  = linspace(0, 1, n);
    v1 = randn(3, 1);
    v2 = randn(3, 1);
    k1 = unifrnd(K1, K2);
    k2 = unifrnd(K1, K2);
    d  = p3 - p0;
    L  = norm(d);
    u  = d / L;
    n1 = v1 - (v1' * u) * u; n1 = n1 / norm(n1);
    n2 = v2 - (v2' * u) * u; n2 = n2 / norm(n2);
    p1 = p0 + alpha1 * d + k1 * L * n1;
    p2 = p0 + (1 - alpha2) * d + k2 * L * n2;

    x = (1 - t).^3 .* p0 + ...
        3 * (1 - t).^2 .* t .* p1 + ...
        3 * (1 - t) .* (t.^2) .* p2 + ...
        (t.^3) .* p3;

    dx = 3 * (1 - t).^2 .* (p1 - p0) + ...
         6 * (1 - t) .* t .* (p2 - p1) + ...
         3 * (t.^2) .* (p3 - p2);

    speed = sqrt(sum(dx.^2, 1));
    T     = dx ./ speed;
end

%% Classic SDP (no rank enforcement beyond relaxation)
function [u_opt, Z, mat] = sdp(x_traj, x_heading, x_angles, N, rho)
    n     = N;
    idx_u = 1:3;
    idx_r = 4:3+n;
    idx_1 = n+4;
    dim   = idx_1;

    C = zeros(dim);
    for i = 1:n
        c_i          = zeros(dim,1);
        c_i(idx_u)   = -sin(x_angles(i)) * x_heading(:,i);
        c_i(idx_r(i))=  cos(x_angles(i));
        c_i(idx_1)   =  sin(x_angles(i)) * (x_heading(:,i)' * x_traj(:,i));
        C = C + c_i * c_i';
    end

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

    u_opt = Z(idx_u, idx_1);
    mat   = Z(idx_u, idx_u) - u_opt * u_opt';
end

%% SDP4 – Penalty-CCP (linearized rank penalty)
function [u_opt, Z, mat] = sdp4(x_traj, x_heading, x_angles, N, rho)
    n     = N;
    idx_u = 1:3;
    idx_r = 4:3+n;
    idx_1 = n+4;
    dim   = idx_1;

    C = zeros(dim);
    for i = 1:n
        c_i          = zeros(dim,1);
        c_i(idx_u)   = -sin(x_angles(i)) * x_heading(:,i);
        c_i(idx_r(i))=  cos(x_angles(i));
        c_i(idx_1)   =  sin(x_angles(i)) * (x_heading(:,i)' * x_traj(:,i));
        C = C + c_i * c_i';
    end

    % ---------- Initial standard SDP ----------
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

    % ---------- Penalty-CCP loop ----------
    max_iter = 15;
    mu       = rho;

    [V, D] = eig(Z);
    [~, idx] = max(diag(D));
    v_k = V(:, idx);
    v_k = v_k / norm(v_k);

    for iter = 1:max_iter
        cvx_begin sdp quiet
            variable Z(dim,dim) symmetric
            minimize( trace(C*Z) + mu*(trace(Z) - trace((v_k*v_k')*Z)) )
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

        [V, D] = eig(Z);
        [~, idx] = max(diag(D));
        v_k = V(:, idx);
        v_k = v_k / norm(v_k);

        %mu = mu * 1.2;
    end

    % Normalize homogeneous coordinate
    if v_k(idx_1) < 0
        v_k = -v_k;
    end
    v_k = v_k / v_k(idx_1);

    u_opt = v_k(idx_u);
    mat   = Z(idx_u, idx_u) - u_opt * u_opt';
end
