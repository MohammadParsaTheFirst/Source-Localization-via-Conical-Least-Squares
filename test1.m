clear, clc;

x_0 = [1; 2; 3]; 
x_1 = [100; 200; 300];
%x_source = [-45; 567; -456];
x_source = [-45; 567; -456];

N = [15, 25, 35, 45, 55];%, 50, 75, 100, 250];
Errors= [];
Ranks = [];


Mats = cell(1, 3);
for j=1:length(N)
    
    n = N(j); rho = 0.000001;
    [x_traj, x_heading] = traj_gen(n, x_0, x_1, 0.15, 0.85, 0.15, 0.85);
    
    % Plot trajectory (x is a 3 x n matrix)
    plot3(x_traj(1,:), x_traj(2,:), x_traj(3,:), 'LineWidth', 2);
    grid on;
    hold on;
    plot3(x_source(1,:), x_source(2,:), x_source(3,:), 'ro', 'LineWidth', 2);
    hold on;
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('Generated 3D Trajectory');
    
    x_angles = calc_angles(x_traj, x_source, x_heading);
    
    
    [u_opt, Z, mat ] = sdp(x_traj, x_heading, x_angles, n, rho);
    plot3(u_opt(1,:), u_opt(2,:), u_opt(3,:), 'go', 'LineWidth', 2);
    text(u_opt(1,:), u_opt(2,:), u_opt(3,:), string(j) , 'FontSize', 15);
    grid on;
    hold on;

    Errors(end+1) = norm(u_opt-x_source);
    Ranks(end+1) = rank(Z);

    E = eig(mat);
    c = cond(mat);
    disp("-----")
    disp(mat)
    disp(norm(mat))
    disp(E)
    disp(c)
    disp("-------------")
end



disp(Errors)
disp("-------")
disp(Ranks)





%% -------------------------------------
% x_heading = 3xN
% angles = 1xN 

function [angles] = calc_angles(x_traj, x_sensor, x_heading)
    
    d_norm = (x_sensor - x_traj) ./ sqrt(sum((x_sensor - x_traj).^2, 1));
    cos_theta = sum(d_norm .* x_heading, 1);
    
    angles = acos(max(-1, min(1, cos_theta)));
end


function [L] = Loss(x_traj, x_heading, x_angles, Z, x_source, rho)
    n = length(x_traj);
    for i=1:n
        c_i = zeros(idx_1,1);
        c_i(idx_u) = -sin(x_angles(i)) * x_heading(:,i);
        c_i(idx_r(i)) = cos(x_angles(i));% i-3
        c_i(idx_1) = sin(x_angles(i)) * (x_heading(:,i)' * x_traj(:,i));
        C = C + c_i * c_i';
    end

    Loss = trace( C * Z) - rho*trace(Z);
end

function [x, T] = traj_gen(n, p0, p3, alpha1, alpha2, K1, K2)
    t = linspace(0, 1, n);
    v1 = randn(3, 1);
    v2 = randn(3, 1);
    k1 = unifrnd(K1, K2);
    k2 = unifrnd(K1, K2);
    d = p3 - p0; 
    L = norm(d); 
    u = d / L;
    
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

    % Unit heading vector T(t) = dx / ||dx||
    speed = sqrt(sum(dx.^2, 1)); % Norm along each column (1 x n)
    T = dx ./ speed;
end


function [u_opt, Z , mat] = sdp(x_traj, x_heading, x_angles, N, rho)

    n=N;
    idx_u = 1:3;
    idx_r = 4:3+n;
    idx_1 = n+4;
    
    
    C = zeros(idx_1, idx_1);
    for i=1:n
        c_i = zeros(idx_1,1);
        c_i(idx_u) = -sin(x_angles(i)) * x_heading(:,i);
        c_i(idx_r(i)) = cos(x_angles(i));% i-3
        c_i(idx_1) = sin(x_angles(i)) * (x_heading(:,i)' * x_traj(:,i));
        C = C + c_i * c_i';
    end

    cvx_begin sdp quiet
    
    variable Z(idx_1, idx_1) symmetric;
    minimize( trace(C * Z) + rho*trace(Z) );
    
    
    subject to
    
    
    Z == semidefinite(idx_1);
    % rank(Z) = 1 ---> relaxed
    Z(idx_1, idx_1) == 1; % Z(n+4,n+4) =1
    for i=1:n
        P_i = eye(3,3) - x_heading(:,i) * x_heading(:,i)';
    
        % >=
        Z(idx_r(i),idx_r(i))==trace(P_i*Z(idx_u,idx_u)) - 2*x_traj(:,i)'*P_i*Z(idx_u,idx_1)
        + x_traj(:,i)'*P_i*x_traj(:,i); % relaxation: Z(i+3, i+3) = r_i^2 >= w_i^T P_i w_i
    
        Z(idx_r(i),idx_1) >= 0;

    end
    cvx_end

    u_opt = Z(idx_u, idx_1);% Z(1:3,n+4)
    mat = Z(idx_u, idx_u) - u_opt * u_opt';

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
