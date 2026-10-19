clear; clc; close all;

% Geometry & experiment settings
u_true   = [-5000; -2000; 4000];          % true source (m)
p0       = [1; 2; 3];
p1       = [100; 200; 300];
N        = 35;                            % fixed number of sensing positions
sigma_deg = [0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 3.0 5.0];  % AOA noise std (deg)
nTrials  = 60;                            % increase for smoother curves
rho      = 1e-3;                          % initial rank-promotion penalty

methods = {'O-SDR', 'O-SDR-rank1', 'RP-SDR', 'GT-IRLS', 'Sun'};
nM = numel(methods);
rmse = zeros(numel(sigma_deg), nM);
crlb = zeros(size(sigma_deg));

%% Monte-Carlo
for si = 1:numel(sigma_deg)
    sig = deg2rad(sigma_deg(si));
    err = nan(nTrials, nM);
    cval = zeros(nTrials,1);

    for k = 1:nTrials
        [s, a] = traj_gen(N, p0, p1, 0.15, 0.85, 0.15, 0.85);
        theta_clean = calc_angles(s, u_true, a);
        theta = theta_clean + sig * randn(N,1);

        % CRLB (geometry-dependent)
        cval(k) = conical_crlb(s, a, u_true, sig);

        % ---- O-SDR ----
        sol = conical_sdr(s, a, theta, ones(N,1));
        if ~isempty(sol)
            err(k,1) = norm(sol.u - u_true);
        end

        % ---- O-SDR + first-order rank-1 ----
        sol1 = osdr_rank1(s, a, theta);
        if ~isempty(sol1)
            err(k,2) = norm(sol1.u - u_true);
        end

        % ---- Rank-promoted SDR ----
        if ~isempty(sol)
            sol_rp = rank_promoted_sdr(s, a, theta, ones(N,1), sol.Z, rho);
            if ~isempty(sol_rp)
                err(k,3) = norm(sol_rp.u - u_true);
            end
        end

        % ---- GT-IRLS ----
        sol_gt = gt_irls(s, a, theta, sig);
        if ~isempty(sol_gt)
            err(k,4) = norm(sol_gt.u - u_true);
        end

        % ---- Sun et al. ----
        sol_sun = sun_et_al(s, a, theta, sig);   % sig already in radians
        if ~isempty(sol_sun)
            err(k,5) = norm(sol_sun.u - u_true);
        end
        % try
        %     Q = sig^2 * eye(N);
        %     [~, u_sun] = SA3DLocLA_SDR(theta(:)', a, s, Q);
        %     err(k,5) = norm(u_sun(:) - u_true);
        % catch
        %     % leave NaN if the official Sun code is not available
        % end
    end

    rmse(si,:) = sqrt(mean(err.^2, 1, 'omitnan'));
    crlb(si)   = mean(cval, 'omitnan');

    fprintf('sigma = %5.3f deg | O-SDR=%7.1f  rank1=%7.1f  RP=%7.1f  GT=%7.1f  Sun=%7.1f  CRLB=%7.1f m\n', ...
        sigma_deg(si), rmse(si,:), crlb(si));
end

%% Log-log plot
figure('Color','w');
markers = {'o','s','^','D','v'};
hold on; grid on;
for m = 1:nM
    loglog(sigma_deg, rmse(:,m), [markers{m} '-'], ...
        'LineWidth',1.8, 'MarkerSize',8, 'DisplayName', methods{m});
end
loglog(sigma_deg, crlb, 'k--', 'LineWidth',2.2, 'DisplayName','CRLB');
set(gca,'XScale','log','YScale','log');
xlabel('\sigma_\theta (deg)');
ylabel('RMSE localization error (m)');
title(sprintf('Conical LS  (N = %d, %d trials)', N, nTrials));
legend('Location','northwest');
hold off;

% Optional: save numbers
save('conical_noise_sweep_results.mat', 'sigma_deg', 'rmse', 'crlb', 'methods', 'N', 'nTrials');


%% ------------------------------------------------------------------
function theta = calc_angles(s, u, a)
% s(:,i) sensing position, u source, a(:,i) unit array axis
N = size(s,2);
theta = zeros(N,1);
for i = 1:N
    v = u - s(:,i);
    theta(i) = acos(max(-1, min(1, a(:,i)'*v / norm(v))));
end
end

function [s, a] = traj_gen(N, p0, p1, alpha1, alpha2, K1, K2)
% Simple random cubic Bezier trajectory + unit tangent orientations
p0 = p0(:); p1 = p1(:);
span = max(norm(p1-p0), 1);
d1 = p0 + alpha1*(p1-p0) + K1*span*randn(3,1);
d2 = p0 + alpha2*(p1-p0) + K2*span*randn(3,1);
t  = linspace(0,1,N);
s  = zeros(3,N); a = zeros(3,N);
for i = 1:N
    q = t(i);
    s(:,i) = (1-q)^3*p0 + 3*(1-q)^2*q*d1 + 3*(1-q)*q^2*d2 + q^3*p1;
    v = 3*(1-q)^2*(d1-p0) + 6*(1-q)*q*(d2-d1) + 3*q^2*(p1-d2);
    if norm(v) < 1e-10, v = p1-p0; end
    a(:,i) = v / norm(v);
end
end
% ------------------------------------------------------------------
function sol = conical_sdr(s, a, theta, w)
% Ordinary SDR for conical least squares (requires CVX + SDPT3/MOSEK)
N = size(s,2); theta = theta(:); w = w(:);
if nargin < 4 || isempty(w), w = ones(N,1); end

center = mean(s,2);
scale  = max(vecnorm(s-center,2,1));
if scale <= eps, error('Sensing positions have zero aperture.'); end
sn = (s - center)/scale;

d = N+4; h = d;
C = zeros(d);
for i = 1:N
    c = zeros(d,1);
    c(1:3)   = -sin(theta(i))*a(:,i);
    c(3+i)   =  cos(theta(i));
    c(h)     =  sin(theta(i))*a(:,i)'*sn(:,i);
    C = C + w(i)*(c*c');
end

cvx_begin quiet
cvx_solver mosek
variable Z(d,d) symmetric
minimize(trace(C*Z))
subject to
Z == semidefinite(d);
Z(h,h) == 1;
for i = 1:N
    P = eye(3) - a(:,i)*a(:,i)';
    Z(3+i,3+i) == trace(P*Z(1:3,1:3)) ...
        - 2*sn(:,i)'*P*Z(1:3,h) + sn(:,i)'*P*sn(:,i);
    Z(3+i,h) >= 0;
    cos(theta(i))*a(:,i)'*(Z(1:3,h) - sn(:,i)*Z(h,h)) ...
        + sin(theta(i))*Z(3+i,h) >= 0;
end
cvx_end

if isempty(Z) || any(isnan(Z(:)))
    sol = []; return;
end

u = center + scale*Z(1:3,h);
e = sort(real(eig((Z+Z')/2)), 'descend');
sol = struct('u',u, 'Z',Z, 'center',center, 'scale',scale, ...
    'objective',cvx_optval, ...
    'rank_gap', max(0,(trace(Z)-e(1))/max(trace(Z),eps)));
end
% ------------------------------------------------------------------
function sol = osdr_rank1(s, a, theta, w)
% O-SDR followed by dominant-eigenvector rank-1 extraction
if nargin < 4, w = ones(size(s,2),1); end
sol = conical_sdr(s, a, theta, w);
if isempty(sol), return; end

Z = (sol.Z + sol.Z')/2;
[V, D] = eig(Z);
[~, idx] = max(real(diag(D)));
v = real(V(:,idx));
if abs(v(end)) < 1e-12
    return;   % keep original u
end
u_scaled = v(1:3) / v(end);
sol.u = sol.center + sol.scale * u_scaled;
sol.method = 'O-SDR-rank1';
end
% ------------------------------------------------------------------
function sol = rank_promoted_sdr(s, a, theta, w, Z0, rho)
% Spectral rank promotion: min tr(CZ) + mu*(tr(Z)-v'*Z*v)
N = size(s,2); theta = theta(:); w = w(:);
center = mean(s,2);
scale  = max(vecnorm(s-center,2,1));
sn = (s-center)/scale;
d = N+4; h = d;

C = zeros(d);
for i = 1:N
    c = zeros(d,1);
    c(1:3) = -sin(theta(i))*a(:,i);
    c(3+i) =  cos(theta(i));
    c(h)   =  sin(theta(i))*a(:,i)'*sn(:,i);
    C = C + w(i)*(c*c');
end

Zk = Z0; mu = rho;
for stage = 1:8
    for it = 1:20
        [V,D] = eig((Zk+Zk')/2);
        [~,ix] = max(real(diag(D)));
        v = real(V(:,ix)); v = v/norm(v);

        cvx_begin quiet
        cvx_solver mosek
        variable Z(d,d) symmetric
        minimize(trace(C*Z) + mu*(trace(Z) - v'*Z*v))
        subject to
        Z == semidefinite(d);
        Z(h,h) == 1;
        for i = 1:N
            P = eye(3)-a(:,i)*a(:,i)';
            Z(3+i,3+i) == trace(P*Z(1:3,1:3)) ...
                -2*sn(:,i)'*P*Z(1:3,h) + sn(:,i)'*P*sn(:,i);
            Z(3+i,h) >= 0;
            cos(theta(i))*a(:,i)'*(Z(1:3,h)-sn(:,i)*Z(h,h)) ...
                + sin(theta(i))*Z(3+i,h) >= 0;
        end
        cvx_end

        if isempty(Z) || any(isnan(Z(:))), break; end
        change = norm(Z-Zk,'fro') / max(1,norm(Zk,'fro'));
        Zk = (Z+Z')/2;
        if change < 1e-5, break; end
    end
    ev = sort(real(eig(Zk)),'descend');
    gap = max(0,(trace(Zk)-ev(1))/max(trace(Zk),eps));
    if gap < 1e-3, break; end
    mu = 3*mu;
end

sol = struct('u', center+scale*Zk(1:3,h), 'Z',Zk, 'rank_gap',gap, ...
    'objective',cvx_optval);
end
% ------------------------------------------------------------------
function sol = gt_irls(s, a, theta, sigma)
% Gap-triggered O-SDR -> spectral rank promotion + range-normalized IRLS
N = size(s,2); theta = theta(:);
base = conical_sdr(s, a, theta, ones(N,1));
if isempty(base), sol = []; return; end

[JA, pass] = angular_test(s, a, theta, base.u, sigma);
if base.rank_gap < 5e-3 && pass
    sol = base; sol.method = 'O-SDR'; sol.J_A = JA; return;
end

u = base.u; cand = base;
for k = 1:5
    R = vecnorm(u-s,2,1)';
    w = 1./(max(R,1e-6).^2 * sigma^2); w = w/mean(w);
    cand = rank_promoted_sdr(s, a, theta, w, cand.Z, 1e-2);
    if isempty(cand), break; end
    if norm(cand.u-u)/max(1,norm(u)) < 1e-3, break; end
    u = cand.u;
end

if isempty(cand)
    sol = base; return;
end

[JA, pass] = angular_test(s, a, theta, cand.u, sigma);
if cand.rank_gap < 1e-3 && pass
    sol = cand; sol.method = 'GT-IRLS'; sol.J_A = JA; return;
end

% Conservative fallback
[JA0,~] = angular_test(s, a, theta, base.u, sigma);
if JA0 <= JA
    sol = base; sol.J_A = JA0;
else
    sol = cand; sol.J_A = JA;
end
sol.method = 'GT-IRLS (candidate selection)';
end

function [J, pass] = angular_test(s, a, theta, u, sigma)
N = size(s,2);
th = calc_angles(s, u, a);
J = sum(((th-theta)/sigma).^2);
% Approximate chi-square gate, N-3 degrees of freedom, 99 %
pass = J <= 2*gammaincinv(0.99, (N-3)/2);
end
% ------------------------------------------------------------------
function rmse_crlb = conical_crlb(s, a, u, sigma)
% RMSE lower bound = sqrt(trace(CRLB)) for conical AOA
% sigma in radians
N = size(s,2);
FIM = zeros(3);
for i = 1:N
    v = u - s(:,i);
    r = norm(v);
    if r < 1e-9, continue; end
    c = max(-1,min(1, a(:,i)'*v / r));
    sth = sqrt(max(1-c^2, 1e-16));
    dcos = a(:,i)/r - c*v/r^2;
    dth  = -dcos / sth;          % 3 x 1
    FIM  = FIM + (dth*dth') / sigma^2;
end
try
    rmse_crlb = sqrt(trace(inv(FIM)));
catch
    rmse_crlb = NaN;
end
end
% ------------------------------------------------------------------
function sol = sun_et_al(s, a, theta, sigma)
% SUN_ET_AL  Sun et al. TSP 2022 – 3-D SA localization (SDR + refinement)
%
%   sol = sun_et_al(s, a, theta, sigma)
%
% Inputs (consistent with your other solvers):
%   s      – 3×N  sensor positions  (or N×3 – auto-detected)
%   a      – 3×N  unit array orientation vectors (or N×3)
%   theta  – N×1  measured space angles (rad)
%   sigma  – scalar noise std (rad).  Q = sigma^2 * I is used.
%            (pass [] or omit if you already have a full Q; see below)
%
% Output:
%   sol.u  – 3×1 source position estimate (refined)
%   sol    – empty if the SDP fails
%
% Reference:
%   Y. Sun, K. C. Ho, L. Gao, J. Zou, Y. Yang, L. Chen,
%   "Three dimensional source localization using arrival angles from
%    linear arrays: Analytical investigation and optimal solution,"
%   IEEE Trans. Signal Process., vol. 70, pp. 1864–1879, 2022.
%
% The implementation follows the authors' official MATLAB code
% (SA3DLocLA_SDR) exactly, only the calling signature is adapted
% to your Monte-Carlo script.

    sol = [];

    % ---- orient matrices to 3×N ---------------------------------
    if size(s,1) ~= 3, s = s.'; end
    if size(a,1) ~= 3, a = a.'; end
    theta = theta(:);
    N = size(s,2);
    if numel(theta) ~= N
        error('theta length must equal number of sensors');
    end

    % ---- noise covariance ---------------------------------------
    if nargin < 4 || isempty(sigma)
        Q = eye(N);                 % unweighted (or supply your own Q)
    else
        Q = (sigma^2) * eye(N);
    end

    % ---- call the core solver (authors' algorithm) --------------
    try
        [uRfn, ~] = SA3DLocLA_SDR_core(theta, a, s, Q);
        if any(~isfinite(uRfn))
            return;
        end
        sol.u = uRfn(:);
    catch
        % SDP failed / CVX not available / numerical issue
        disp(" SDP failed / CVX not available / numerical issue");
        sol = [];
    end
end


% =================================================================
%  Core routine – verbatim from Sun & Ho's official code
%  (only minor robustness / dimension checks added)
% =================================================================
function [uRfn, uSDR] = SA3DLocLA_SDR_core(theta, gamma, S, Q)

    [Ndim, M] = size(S);          % Ndim = 3, M = #sensors
    if Ndim ~= 3
        error('S must be 3×M');
    end

    h = diag(gamma' * S);         % M×1
    G = [gamma', -diag(cos(theta))];   % M×(3+M)
    B = eye(M);

    % ---- two iterations of weighted SDR -------------------------
    for itr = 1:2
        W = inv(B * Q * B);
        A = [G'*W*G,  -G'*W*h;
            -h'*W*G,   h'*W*h];

        cvx_begin quiet
            cvx_solver sdpt3
            variable P(M+Ndim, M+Ndim) symmetric
            variable p(M+Ndim, 1)
            SM = [P, p; p', 1];
            minimize( trace(SM * A) )
            subject to
                for i = 1:M
                    P(i+Ndim, i+Ndim) == ...
                        trace(P(1:Ndim,1:Ndim)) - 2*S(:,i)'*p(1:Ndim) + S(:,i)'*S(:,i);
                    p(i+Ndim) >= 0;
                end
                for i = 1:M-1
                    for j = i+1:M
                        P(i+Ndim, j+Ndim) >= abs( ...
                            trace(P(1:Ndim,1:Ndim)) ...
                            - S(:,i)'*p(1:Ndim) - S(:,j)'*p(1:Ndim) ...
                            + S(:,i)'*S(:,j) );
                    end
                end
                SM == semidefinite(M + Ndim + 1);
        cvx_end

        if ~strcmp(cvx_status, 'Solved') && ~strcmp(cvx_status, 'Inaccurate/Solved')
            uSDR = nan(3,1);
            uRfn = nan(3,1);
            return;
        end

        u2   = sqrt(diag(SM(1:Ndim,1:Ndim)));
        uSDR = func_findu(theta, gamma, S, Q, u2, M);

        r = sqrt(sum((S - uSDR).^2, 1)).';
        B = -diag(r .* sin(theta));
    end

    % ---- algebraic refinement (Gauss-Newton style) --------------
    uRfn = uSDR;
    for iter = 1:2
        r = sqrt(sum((S - uRfn).^2, 1)).';
        g = h - gamma'*uRfn + diag(cos(theta))*r;

        Ht = zeros(Ndim+M, Ndim);
        Ht(1:Ndim,1:Ndim) = eye(Ndim);
        for i = 1:M
            Ht(Ndim+i,:) = (uRfn - S(:,i))' / norm(uRfn - S(:,i));
        end
        H2  = G * Ht;
        du2 = -(H2'*W*H2) \ (H2'*W*g);
        uRfn = uRfn - du2;

        r = sqrt(sum((S - uSDR).^2, 1)).';   % keep weight from SDR stage
        B = -diag(r .* sin(theta));
        W = inv(B * Q * B);
    end
end


% ---- sign-ambiguity resolution (authors' helper) ---------------
function uu = func_findu(theta, gamma, S, Q, pos, M)
    sgn = [ 1  1  1;
            1  1 -1;
            1 -1  1;
            1 -1 -1;
           -1  1  1;
           -1  1 -1;
           -1 -1  1;
           -1 -1 -1]';
    J = zeros(8,1);
    for j = 1:8
        u = sgn(:,j) .* pos;
        est_theta = zeros(M,1);
        for i = 1:M
            est_theta(i) = acos( gamma(:,i)'*(u-S(:,i)) / norm(u-S(:,i)) );
        end
        J(j) = (theta - est_theta)' / Q * (theta - est_theta);
    end
    [~, ind] = min(J);
    uu = sgn(:,ind) .* pos;
end
