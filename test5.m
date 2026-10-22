%MONTE_CARLO_PROTOCOL_V4  Publication-oriented benchmark for conical localization
% Includes: SDR, RankPromoted, RankPromoted_IRLS, AngularMLE, SunEtAl (TSP 2022)
%
% Design goals (unchanged from v3 + Sun method):
%  * common trajectories and common Gaussian draws across all methods;
%  * separate rank-promotion from WLS reweighting;
%  * report CVX failures, runtime, rank gap, data-fit cost;
%  * direct angular Gaussian MLE/WLS refinement + CRLB;
%  * explicit boundary-event counting (no silent clipping);
%  * all plots in log-log / semi-log scale.
%
% PARALLEL VERSION (v4):
%  - Uses parfor on the independent Monte-Carlo trials.
%  - Requires Parallel Computing Toolbox.
%  - Start a pool once (e.g. parpool('local',20) for your 20-core CPU).
%  - Logic, results and reproducibility are identical to sequential v3.
%
% Start with nTrials=20-50 for debugging.  For a paper use >=500 (preferably 1000-2000).
clear; clc; close all;


% ----- Start parallel pool (uses all 20 cores) -----
if isempty(gcp('nocreate'))
    parpool('local', 20);   % or just parpool if you prefer auto
end
% --------------------------------------------------

rng(10);
N           = 20;
nTrials     = 100;                       % paper: >=500
sigma_deg_grid = [0.05 0.1 0.25 0.5 1 2];
range_grid  = [1e3 2e3 5e3 1e4 2e4];
R0          = 5000;
sigma_range = deg2rad(0.5);
p0 = [1;2;3]; p3 = [100;200;300];
source_dir  = [-5;-2;4]; source_dir = source_dir/norm(source_dir);
method_names = {'SDR','RankPromoted','RankPromoted_IRLS','AngularMLE','SunEtAl'};
M = numel(method_names);

% ----------------------------------------------------------------------
% Pre-generate trajectories and standard-normal draws (common to all methods)
% ----------------------------------------------------------------------
S = cell(nTrials,1); A = cell(nTrials,1); Znoise = cell(nTrials,1);
for t = 1:nTrials
    [S{t},A{t}] = bezier_trajectory(N,p0,p3,0.15,0.85,0.15,0.85);
    Znoise{t} = randn(1,N);
end

% ======================================================================
% A. Noise sweep at fixed source range
% ======================================================================
err2_noise      = nan(numel(sigma_deg_grid),nTrials,M);
runtime_noise   = nan(numel(sigma_deg_grid),nTrials,M);
base_gap_noise  = nan(numel(sigma_deg_grid),nTrials);
rp_gap_noise    = nan(numel(sigma_deg_grid),nTrials);
base_fit_noise  = nan(numel(sigma_deg_grid),nTrials);
rp_fit_noise    = nan(numel(sigma_deg_grid),nTrials);
crlb2_noise     = nan(numel(sigma_deg_grid),nTrials);
boundary_events_noise = zeros(numel(sigma_deg_grid),1);
fail_noise      = zeros(numel(sigma_deg_grid),M);

for is = 1:numel(sigma_deg_grid)
    sigma = deg2rad(sigma_deg_grid(is));
    
    % Temporary storage for reduction variables inside parfor
    be_tmp   = zeros(nTrials,1);
    fail_tmp = zeros(nTrials,M);
    
    parfor t = 1:nTrials
        s = S{t}; a = A{t};
        u_true = mean(s,2) + R0*source_dir;
        th0 = calc_space_angles(s,u_true,a);
        [th,boundary_used] = apply_common_noise(th0,sigma,Znoise{t});
        be_tmp(t) = boundary_used;
        
        err2_t   = nan(1,M);
        runtime_t = nan(1,M);
        base_gap_t = NaN;
        rp_gap_t   = NaN;
        base_fit_t = NaN;
        rp_fit_t   = NaN;
        crlb2_t    = NaN;
        fail_t     = zeros(1,M);
        
        % 1) Ordinary SDR
        try
            tt = tic;
            base = conical_sdr_baseline(s,a,th,ones(1,N));
            runtime_t(1) = toc(tt);
            err2_t(1) = norm(base.u - u_true)^2;
            base_gap_t = base.rank_gap;
            base_fit_t = base.datafit;
        catch ME
            fail_t(1) = 1;
            fprintf(2,'Noise %.2f trial %d SDR failed: %s\n',sigma_deg_grid(is),t,ME.message);
        end
        
        % 2) Rank promotion only
        try
            tt = tic;
            rp0 = conical_rank1_estimator(s,a,th,sigma,struct('reweight_iters',1));
            runtime_t(2) = toc(tt);
            err2_t(2) = norm(rp0.u - u_true)^2;
        catch ME
            fail_t(2) = 1;
            fprintf(2,'Noise %.2f trial %d RP failed: %s\n',sigma_deg_grid(is),t,ME.message);
        end
        
        % 3) Rank promotion + one range-based WLS update
        try
            tt = tic;
            rpw = conical_rank1_estimator(s,a,th,sigma,struct('reweight_iters',2));
            runtime_t(3) = toc(tt);
            err2_t(3) = norm(rpw.u - u_true)^2;
            rp_gap_t = rpw.rank_gap;
            rp_fit_t = rpw.datafit;
        catch ME
            fail_t(3) = 1;
            fprintf(2,'Noise %.2f trial %d RP-WLS failed: %s\n',sigma_deg_grid(is),t,ME.message);
        end
        
        % 4) Direct angular Gaussian MLE/WLS refinement
        try
            tt = tic;
            mle = conical_angular_mle_refine(s,a,th,sigma,rpw.u);   %#ok<NASGU>
            runtime_t(4) = toc(tt);
            err2_t(4) = norm(mle.u - u_true)^2;
        catch ME
            fail_t(4) = 1;
            fprintf(2,'Noise %.2f trial %d MLE failed: %s\n',sigma_deg_grid(is),t,ME.message);
        end
        
        % 5) Sun et al. (IEEE TSP 2022) – SDR + algebraic refinement
        try
            tt = tic;
            sun = sun_et_al(s,a,th,sigma);
            runtime_t(5) = toc(tt);
            if ~isempty(sun) && isfield(sun,'u') && all(isfinite(sun.u))
                err2_t(5) = norm(sun.u - u_true)^2;
            else
                fail_t(5) = 1;
            end
        catch ME
            fail_t(5) = 1;
            fprintf(2,'Noise %.2f trial %d SunEtAl failed: %s\n',sigma_deg_grid(is),t,ME.message);
        end
        
        cr = conical_crlb(u_true,s,a,sigma);
        crlb2_t = cr.rmse_bound^2;
        
        % Write results back (sliced variables)
        err2_noise(is,t,:)    = err2_t;
        runtime_noise(is,t,:) = runtime_t;
        base_gap_noise(is,t)  = base_gap_t;
        rp_gap_noise(is,t)    = rp_gap_t;
        base_fit_noise(is,t)  = base_fit_t;
        rp_fit_noise(is,t)    = rp_fit_t;
        crlb2_noise(is,t)     = crlb2_t;
        fail_tmp(t,:)         = fail_t;
    end
    
    boundary_events_noise(is) = sum(be_tmp);
    fail_noise(is,:) = sum(fail_tmp,1);
    fprintf('Noise %.2f deg finished.\n',sigma_deg_grid(is));
end

[RMSE_noise,CIlo_noise,CIhi_noise] = summarize_rmse(err2_noise,1000);
CRLB_noise = sqrt(mean(crlb2_noise,2,'omitnan'));
MedRuntime_noise = squeeze(median(runtime_noise,2,'omitnan'));
MedBaseGap_noise = median(base_gap_noise,2,'omitnan');
MedRPGap_noise   = median(rp_gap_noise,2,'omitnan');

% ======================================================================
% B. Range sweep at fixed angular noise
% ======================================================================
err2_range     = nan(numel(range_grid),nTrials,M);
runtime_range  = nan(numel(range_grid),nTrials,M);
base_gap_range = nan(numel(range_grid),nTrials);
rp_gap_range   = nan(numel(range_grid),nTrials);
crlb2_range    = nan(numel(range_grid),nTrials);
boundary_events_range = zeros(numel(range_grid),1);
fail_range     = zeros(numel(range_grid),M);

for ir = 1:numel(range_grid)
    be_tmp   = zeros(nTrials,1);
    fail_tmp = zeros(nTrials,M);
    
    parfor t = 1:nTrials
        s = S{t}; a = A{t};
        u_true = mean(s,2) + range_grid(ir)*source_dir;
        th0 = calc_space_angles(s,u_true,a);
        [th,boundary_used] = apply_common_noise(th0,sigma_range,Znoise{t});
        be_tmp(t) = boundary_used;
        
        err2_t    = nan(1,M);
        runtime_t = nan(1,M);
        base_gap_t = NaN;
        rp_gap_t   = NaN;
        crlb2_t    = NaN;
        fail_t     = zeros(1,M);
        
        try
            tt = tic; base = conical_sdr_baseline(s,a,th,ones(1,N));
            runtime_t(1) = toc(tt);
            err2_t(1) = norm(base.u-u_true)^2;
            base_gap_t = base.rank_gap;
        catch
            fail_t(1) = 1;
        end
        
        try
            tt = tic; rp0 = conical_rank1_estimator(s,a,th,sigma_range,struct('reweight_iters',1));
            runtime_t(2) = toc(tt);
            err2_t(2) = norm(rp0.u-u_true)^2;
        catch
            fail_t(2) = 1;
        end
        
        try
            tt = tic; rpw = conical_rank1_estimator(s,a,th,sigma_range,struct('reweight_iters',2));
            runtime_t(3) = toc(tt);
            err2_t(3) = norm(rpw.u-u_true)^2;
            rp_gap_t = rpw.rank_gap;
        catch
            fail_t(3) = 1;
        end
        
        try
            tt = tic; mle = conical_angular_mle_refine(s,a,th,sigma_range,rpw.u);
            runtime_t(4) = toc(tt);
            err2_t(4) = norm(mle.u-u_true)^2;
        catch
            fail_t(4) = 1;
        end
        
        % 5) Sun et al.
        try
            tt = tic; sun = sun_et_al(s,a,th,sigma_range);
            runtime_t(5) = toc(tt);
            if ~isempty(sun) && isfield(sun,'u') && all(isfinite(sun.u))
                err2_t(5) = norm(sun.u-u_true)^2;
            else
                fail_t(5) = 1;
            end
        catch
            fail_t(5) = 1;
        end
        
        cr = conical_crlb(u_true,s,a,sigma_range);
        crlb2_t = cr.rmse_bound^2;
        
        err2_range(ir,t,:)    = err2_t;
        runtime_range(ir,t,:) = runtime_t;
        base_gap_range(ir,t)  = base_gap_t;
        rp_gap_range(ir,t)    = rp_gap_t;
        crlb2_range(ir,t)     = crlb2_t;
        fail_tmp(t,:)         = fail_t;
    end
    
    boundary_events_range(ir) = sum(be_tmp);
    fail_range(ir,:) = sum(fail_tmp,1);
    fprintf('Range %g finished.\n',range_grid(ir));
end

[RMSE_range,CIlo_range,CIhi_range] = summarize_rmse(err2_range,1000);
CRLB_range = sqrt(mean(crlb2_range,2,'omitnan'));
MedRuntime_range = squeeze(median(runtime_range,2,'omitnan'));
MedBaseGap_range = median(base_gap_range,2,'omitnan');
MedRPGap_range   = median(rp_gap_range,2,'omitnan');

% ======================================================================
% Console summary
% ======================================================================
midN = ceil(numel(sigma_deg_grid)/2);
fprintf('\nMedian runtime at sigma=%.2f deg:\n',sigma_deg_grid(midN));
for m = 1:M
    fprintf('  %-20s %.4g s\n',method_names{m},MedRuntime_noise(midN,m));
end
base_err = sqrt(reshape(err2_noise(:,:,1),[],1));
valid = isfinite(base_gap_noise(:)) & isfinite(base_err);
x = log10(base_gap_noise(valid)+1e-16);
y = log10(base_err(valid));
if numel(x)>3
    Rxy = corrcoef(x,y); gap_error_corr = Rxy(1,2);
else
    gap_error_corr = NaN;
end
fprintf('Gap/error log-correlation for ordinary SDR: %.4f\n',gap_error_corr);
fprintf('Boundary redraw counts (noise sweep): '); fprintf('%d ',boundary_events_noise); fprintf('\n');
fprintf('Failure counts by noise point/method:\n'); disp(fail_noise);
fprintf('Failure counts by range point/method:\n'); disp(fail_range);

% ======================================================================
% Save everything
% ======================================================================
fname = sprintf('monte_carlo_protocol_v4_N=%d_n_trial=%d.mat',N,nTrials);
save(fname, ...
'RMSE_noise','CIlo_noise','CIhi_noise','CRLB_noise', ...
'RMSE_range','CIlo_range','CIhi_range','CRLB_range', ...
'sigma_deg_grid','range_grid', ...
'err2_noise','err2_range','runtime_noise','runtime_range', ...
'MedRuntime_noise','MedRuntime_range', ...
'base_gap_noise','rp_gap_noise','base_gap_range','rp_gap_range', ...
'base_fit_noise','rp_fit_noise', ...
'fail_noise','fail_range', ...
'boundary_events_noise','boundary_events_range', ...
'gap_error_corr','N','nTrials','method_names','R0','sigma_range');
fprintf('Results saved to %s\n',fname);

% ======================================================================
% All plots – forced log scale
% ======================================================================
colors = lines(M);
% 1. Noise sweep – RMSE (log-log)
figure('Name','Noise Sweep – RMSE','Color','w'); hold on; grid on; box on;
for m = 1:M
    loglog(sigma_deg_grid, RMSE_noise(:,m), '-o', ...
'LineWidth',1.6, 'MarkerSize',7, 'Color',colors(m,:), ...
'DisplayName',method_names{m});
end
loglog(sigma_deg_grid, CRLB_noise, 'k--', 'LineWidth',1.8, 'DisplayName','CRLB');
xlabel('Angle-noise standard deviation [deg]');
ylabel('Position RMSE');
title(sprintf('Noise sweep (N=%d, R=%g)',N,R0));
legend('Location','northwest');
xlim([min(sigma_deg_grid)*0.8 max(sigma_deg_grid)*1.2]);
% 2. Noise sweep – Median rank gap (log-log)
figure('Name','Noise Sweep – Rank Gap','Color','w');
loglog(sigma_deg_grid, MedBaseGap_noise, '-o', 'LineWidth',1.6, 'MarkerSize',7, ...
'DisplayName','Ordinary SDR'); hold on; grid on; box on;
loglog(sigma_deg_grid, MedRPGap_noise, '-s', 'LineWidth',1.6, 'MarkerSize',7, ...
'DisplayName','Rank-promoted + IRLS');
xlabel('Angle-noise standard deviation [deg]');
ylabel('Median normalized rank gap');
title('Relaxation gap vs noise level');
legend('Location','best');
% 3. Gap vs localization error (log-log scatter)
figure('Name','Gap vs Error','Color','w');
loglog(base_gap_noise(valid)+1e-16, base_err(valid), 'o', ...
'MarkerSize',5, 'MarkerFaceColor',[0.2 0.4 0.8]);
grid on; box on;
xlabel('Ordinary-SDR normalized rank gap');
ylabel('SDR position error');
title(sprintf('Relaxation gap vs localization error (corr(log)=%.3f)',gap_error_corr));
% 4. Range sweep – RMSE (log-log)
figure('Name','Range Sweep – RMSE','Color','w'); hold on; grid on; box on;
for m = 1:M
    loglog(range_grid, RMSE_range(:,m), '-o', ...
'LineWidth',1.6, 'MarkerSize',7, 'Color',colors(m,:), ...
'DisplayName',method_names{m});
end
loglog(range_grid, CRLB_range, 'k--', 'LineWidth',1.8, 'DisplayName','CRLB');
xlabel('Source range');
ylabel('Position RMSE');
title(sprintf('Range sweep (N=%d, \\sigma=%.2f deg)',N,rad2deg(sigma_range)));
legend('Location','northwest');
xlim([min(range_grid)*0.8 max(range_grid)*1.2]);
% 5. Median runtime comparison (noise sweep, log-y)
figure('Name','Median Runtime (Noise Sweep)','Color','w');
semilogy(sigma_deg_grid, MedRuntime_noise, '-o', 'LineWidth',1.5, 'MarkerSize',6);
grid on; box on;
xlabel('Angle-noise standard deviation [deg]');
ylabel('Median runtime [s]');
title('Median runtime vs noise level');
legend(method_names,'Location','best');
set(gca,'XScale','log');



%%







% ----------------------------------------------------------------------
% Local helper functions 
% ----------------------------------------------------------------------
function [th,nredraw] = apply_common_noise(th0,sigma,z)
    th = th0 + sigma*z;
    bad = (th<0 | th>pi);
    nredraw = sum(bad);
    if any(bad)
        tmp = sample_noisy_angles(th0(bad),sigma,'reject');
        th(bad) = tmp;
    end
end

function [rmse,lo,hi] = summarize_rmse(err2,nboot)
    [K,~,M] = size(err2);
    rmse = nan(K,M); lo = nan(K,M); hi = nan(K,M);
    for k = 1:K
        for m = 1:M
            x = squeeze(err2(k,:,m)); x = x(isfinite(x));
            if isempty(x), continue; end
            rmse(k,m) = sqrt(mean(x));
            if numel(x)>=5
                b = zeros(nboot,1); n = numel(x);
                for j = 1:nboot
                    id = randi(n,n,1); b(j) = sqrt(mean(x(id)));
                end
                b = sort(b);
                lo(k,m) = b(max(1,round(0.025*nboot)));
                hi(k,m) = b(min(nboot,round(0.975*nboot)));
            end
        end
    end
end

function [x,T] = bezier_trajectory(n,p0,p3,alpha1,alpha2,K1,K2)
    tt = linspace(0,1,n);
    v1 = randn(3,1); v2 = randn(3,1);
    k1 = K1+(K2-K1)*rand; k2 = K1+(K2-K1)*rand;
    d = p3-p0; L = norm(d); e = d/L;
    n1 = v1-(v1'*e)*e; n1 = n1/norm(n1);
    n2 = v2-(v2'*e)*e; n2 = n2/norm(n2);
    p1 = p0+alpha1*d+k1*L*n1;
    p2 = p0+(1-alpha2)*d+k2*L*n2;
    x = (1-tt).^3.*p0 + 3*(1-tt).^2.*tt.*p1 + 3*(1-tt).*tt.^2.*p2 + tt.^3.*p3;
    dx = 3*(1-tt).^2.*(p1-p0) + 6*(1-tt).*tt.*(p2-p1) + 3*tt.^2.*(p3-p2);
    T = dx./sqrt(sum(dx.^2,1));
end

% =====================================================================
% CRLB Estimator
% =====================================================================
function out = conical_crlb(u, s, a, sigma_theta)
%CONICAL_CRLB Fisher information / position CRLB for independent angle noise.
%
% Model: theta_i(u)=acos(a_i' q_i), q_i=(u-s_i)/||u-s_i||.
% The gradient is
%   grad theta_i = -(I-q_i q_i') a_i / (R_i sin(theta_i)).
%
% out.covariance = pinv(FIM)
% out.rmse_bound = sqrt(trace(out.covariance))

    n=size(s,2);
    if isscalar(sigma_theta), sigma_theta=repmat(sigma_theta,1,n); end
    sigma_theta=sigma_theta(:).';
    an=sqrt(sum(a.^2,1)); a=a./an;

    F=zeros(3);
    grads=zeros(3,n);
    theta=zeros(1,n);
    for i=1:n
        w=u(:)-s(:,i);
        R=norm(w);
        q=w/R;
        beta=max(-1,min(1,a(:,i)'*q));
        theta(i)=acos(beta);
        st=max(sqrt(max(0,1-beta^2)),1e-10);
        g=-(eye(3)-q*q')*a(:,i)/(R*st);
        grads(:,i)=g;
        F=F+(1/sigma_theta(i)^2)*(g*g');
    end
    Cov=pinv(F);
    out=struct('fim',F,'covariance',Cov,'rmse_bound',sqrt(trace(Cov)), ...
               'gradients',grads,'theta',theta,'condition',cond(F));
end


% =====================================================================
% Gap-triggered Rank Promotion Estimator (Gt-RP-IRLS)
% =====================================================================
function out = gt_rp_irls_v5(theta,A,S,sigma_theta,opts)
%GT_RP_IRLS_V5 Gap-triggered rank-promotion estimator used by paper v5/v5.1.
%
% Inputs
%   theta       N-by-1 measured space angles [rad], measured from +array axis
%   A           3-by-N unit array-axis vectors
%   S           3-by-N array reference positions [m]
%   sigma_theta scalar or N-by-1 angular standard deviation [rad]
%   opts        optional structure
%
% Main defaults matching the manuscript:
%   opts.gap_threshold       = 5e-3
%   opts.rank_tight_threshold= 1e-3
%   opts.alpha               = 0.01
%   opts.solver              = 'sdpt3'
%   opts.irls_passes         = 2
%
% Logic
%   1) Solve ordinary branch-aware SDR.
%   2) Accept it only when the normalized spectral gap is below the gate
%      threshold and its angular chi-square cost is measurement-consistent.
%   3) Otherwise run spectral rank-promotion + IRLS.
%   4) If that repaired solution is not rank-tight/data-consistent, run the
%      Schur-slack rank-promotion fallback.
%   5) Select the repaired candidate with the smaller angular WLS cost.
%
% Requires CVX and an SDP-capable solver.

if nargin < 5 || isempty(opts), opts = struct; end
opts = def(opts,'gap_threshold',5e-3);
opts = def(opts,'rank_tight_threshold',1e-3);
opts = def(opts,'alpha',0.01);
opts = def(opts,'solver','sdpt3');
opts = def(opts,'irls_passes',2);
opts = def(opts,'mu0_factor',1e-3);
opts = def(opts,'mu_growth',5);
opts = def(opts,'max_stages',4);
opts = def(opts,'max_mm',5);
opts = def(opts,'mm_tol',1e-4);

N = size(S,2);
theta = theta(:);
if size(A,1)~=3 || size(S,1)~=3 || size(A,2)~=N || numel(theta)~=N
    error('A and S must be 3-by-N and theta must contain N angles.');
end
A = A ./ vecnorm(A,2,1);
if isscalar(sigma_theta), sigma_theta = repmat(sigma_theta,N,1); else, sigma_theta=sigma_theta(:); end
if numel(sigma_theta)~=N || any(sigma_theta<=0), error('Invalid sigma_theta.'); end

dof = max(N-3,1);
Talpha = 2*gammaincinv(1-opts.alpha,dof/2); % chi-square quantile

% Ordinary SDR.
[u0,Z0,~,status0] = solve_osdr(theta,A,S,ones(N,1),opts.solver);
gap0 = spectral_gap(Z0);
J0 = angular_cost(u0,theta,A,S,sigma_theta);
accept0 = gap0 <= opts.gap_threshold && J0 <= Talpha;

out = struct;
out.u_osdr=u0; out.Z_osdr=Z0; out.gap_osdr=gap0; out.J_osdr=J0;
out.cvx_status_osdr=status0; out.Talpha=Talpha; out.triggered=~accept0;
out.used_schur=false;

if accept0
    out.u=u0; out.Z=Z0; out.method='O-SDR accepted'; out.J=J0; out.rank_gap=gap0;
    return;
end

% First rescue path: spectral rank promotion + IRLS.
[usp,Zsp,wsp,statussp] = solve_spectral_irls(theta,A,S,sigma_theta,opts);
gapsp = spectral_gap(Zsp);
Jsp = angular_cost(usp,theta,A,S,sigma_theta);
out.u_spectral=usp; out.Z_spectral=Zsp; out.gap_spectral=gapsp; out.J_spectral=Jsp;
out.cvx_status_spectral=statussp;

if gapsp <= opts.rank_tight_threshold && Jsp <= Talpha
    out.u=usp; out.Z=Zsp; out.method='Spectral RP-IRLS'; out.J=Jsp; out.rank_gap=gapsp;
    out.weights=wsp;
    return;
end

% Fallback: Schur-slack rank promotion + IRLS.
out.used_schur=true;
[usc,Zsc,wsc,statussc] = solve_schur_irls(theta,A,S,sigma_theta,opts);
gapsc = spectral_gap(Zsc);
Jsc = angular_cost(usc,theta,A,S,sigma_theta);
out.u_schur=usc; out.Z_schur=Zsc; out.gap_schur=gapsc; out.J_schur=Jsc;
out.cvx_status_schur=statussc;

% Measurement-domain selection. Rank is structural; the measured-angle
% likelihood determines which repaired rank-one candidate is retained.
if Jsc < Jsp
    out.u=usc; out.Z=Zsc; out.method='Schur RP-IRLS fallback'; out.J=Jsc; out.rank_gap=gapsc; out.weights=wsc;
else
    out.u=usp; out.Z=Zsp; out.method='Spectral RP-IRLS after fallback comparison'; out.J=Jsp; out.rank_gap=gapsp; out.weights=wsp;
end
end

function [u,Z,w,status] = solve_osdr(theta,A,S,w,solver)
[Sn,cen,L] = normgeom(S);
[~,Z,status] = cvx_sub(theta,A,Sn,w,[],solver);
u = cen + L*Z(1:3,end);
end

function [u,Z,w,status] = solve_spectral_irls(theta,A,S,sigma,opts)
N=numel(theta); w=ones(N,1); status='';
for pass=1:opts.irls_passes
    [Sn,cen,L]=normgeom(S); [C,Z,status]=cvx_sub(theta,A,Sn,w,[],opts.solver);
    mu=opts.mu0_factor*max(norm(C,2),1e-12);
    for st=1:opts.max_stages
        for it=1:opts.max_mm
            [V,D]=eig((Z+Z')/2); [~,ix]=max(real(diag(D))); q=real(V(:,ix)); q=q/norm(q);
            H=C+mu*(eye(size(Z))-q*q');
            [~,Zn,status]=cvx_sub(theta,A,Sn,w,H,opts.solver);
            rel=norm(Zn-Z,'fro')/max(1,norm(Z,'fro')); Z=Zn;
            if rel<opts.mm_tol, break; end
        end
        if spectral_gap(Z)<opts.rank_tight_threshold, break; end
        mu=opts.mu_growth*mu;
    end
    u=cen+L*Z(1:3,end);
    if pass<opts.irls_passes
        R=vecnorm(u-S,2,1)'; w=1./max((R.*sigma).^2,1e-18); w=w/mean(w);
    end
end
end

function [u,Z,w,status] = solve_schur_irls(theta,A,S,sigma,opts)
N=numel(theta); w=ones(N,1); status='';
for pass=1:opts.irls_passes
    [Sn,cen,L]=normgeom(S); [C,Z,status]=cvx_sub(theta,A,Sn,w,[],opts.solver);
    mu=opts.mu0_factor*max(norm(C,2),1e-12); h=size(Z,1)-1;
    for st=1:opts.max_stages
        for it=1:opts.max_mm
            z=Z(1:h,end); Hpen=zeros(size(Z)); Hpen(1:h,1:h)=eye(h);
            Hpen(1:h,end)=-z; Hpen(end,1:h)=-z'; H=C+mu*Hpen;
            [~,Zn,status]=cvx_sub(theta,A,Sn,w,H,opts.solver);
            rel=norm(Zn-Z,'fro')/max(1,norm(Z,'fro')); Z=Zn;
            if rel<opts.mm_tol, break; end
        end
        X=Z(1:h,1:h); z=Z(1:h,end);
        sg=max(real(trace(X)-z'*z),0)/max(real(trace(X)),1e-30);
        if sg<opts.rank_tight_threshold, break; end
        mu=opts.mu_growth*mu;
    end
    u=cen+L*Z(1:3,end);
    if pass<opts.irls_passes
        R=vecnorm(u-S,2,1)'; w=1./max((R.*sigma).^2,1e-18); w=w/mean(w);
    end
end
end

function [C,Z,status]=cvx_sub(theta,A,S,w,Hoverride,solver)
N=size(S,2); n=N+4; h=n; C=zeros(n);
for i=1:N
    ci=zeros(n,1); ci(1:3)=-sin(theta(i))*A(:,i); ci(3+i)=cos(theta(i));
    ci(h)=sin(theta(i))*(A(:,i)'*S(:,i)); C=C+w(i)*(ci*ci');
end
if isempty(Hoverride), H=C; else, H=(Hoverride+Hoverride')/2; end
cvx_begin sdp quiet
    cvx_solver(solver)
    cvx_precision high
    variable Z(n,n) symmetric
    minimize(trace(H*Z))
    subject to
        Z >= 0;
        Z(h,h)==1;
        for i=1:N
            ri=3+i; s=S(:,i); a=A(:,i); P=eye(3)-a*a';
            Z(ri,ri)==trace(P*Z(1:3,1:3))-2*s'*P*Z(1:3,h)+s'*P*s;
            Z(ri,h)>=0;
            cos(theta(i))*a'*(Z(1:3,h)-s)+sin(theta(i))*Z(ri,h)>=0;
        end
cvx_end
status=cvx_status;
end

function J=angular_cost(u,theta,A,S,sigma)
D=u-S; R=vecnorm(D,2,1); c=sum(A.*D,1)./R; c=max(-1,min(1,c)); th=acos(c)';
J=sum(((th-theta)./sigma).^2);
end

function g=spectral_gap(Z)
ev=eig((Z+Z')/2); g=real(trace(Z)-max(real(ev)))/max(real(trace(Z)),1e-30); g=max(g,0);
end

function [Sn,cen,L]=normgeom(S)
cen=mean(S,2); Sc=S-cen; L=max(vecnorm(Sc,2,1)); if L<1e-12, L=1; end; Sn=Sc/L;
end

function s=def(s,name,val)
if ~isfield(s,name) || isempty(s.(name)), s.(name)=val; end
end


% =====================================================================
% O-SDR + rank 1 estimation of Z 
% =====================================================================
function result = conical_rank1_estimator(s, a, theta, sigma_theta, opts)
%CONICAL_RANK1_ESTIMATOR Rank-promoted SDR estimator for 3-D conical localization.
%
% Model
% -----
% s(:,i)      : array/sensor reference position in R^3
% a(:,i)      : unit vector along the oriented linear-array axis
% theta(i)    : measured space angle from +a(:,i), in [0, pi]
% sigma_theta : angle standard deviations [rad], scalar or 1-by-N
%
% The physical direction-cosine convention is
%
%     cos(theta_i) = a_i' * (u-s_i)/||u-s_i||.
%
% Define
%     d_i = a_i' * (u-s_i)
%     r_i = ||(I-a_i*a_i')*(u-s_i)|| >= 0.
%
% For the one-nappe cone ray, the local 2-D generator is
%     [cos(theta_i); sin(theta_i)].
% When the projection parameter
%     tau_i = d_i*cos(theta_i) + r_i*sin(theta_i) >= 0,
% the Euclidean distance to the cone surface equals
%     |r_i*cos(theta_i) - d_i*sin(theta_i)|.
%
% This function lifts v=[u; r_1; ...; r_N; 1] to Z=v*v' and uses:
%  (1) a standard SDR;
%  (2) a fixed-mu MM/CCP loop for tr(Z)-lambda_max(Z);
%  (3) continuation in mu only between converged MM stages;
%  (4) optional outer reweighting matched to Gaussian angle noise.
%
% IMPORTANT
% ---------
% This code requires CVX for MATLAB with an SDP-capable solver.
% It has been written to match the corrected mathematical formulation but
% was not executed in this environment because CVX/MATLAB are unavailable.
%
% Output fields include:
%   .u                 position estimate in original coordinates
%   .Z                 final lifted matrix in normalized coordinates
%   .u_local           position estimate in normalized local coordinates
%   .weights           final normalized WLS weights
%   .rank_gap          (tr(Z)-lambda1)/tr(Z)
%   .lambda2_over_1    lambda2/lambda1
%   .history           MM/continuation diagnostics
%   .center, .scale    normalization transform
%   .cvx_status        final CVX status
%
% Recommended calling pattern:
%   opts = struct;
%   opts.reweight_iters = 2;
%   out = conical_rank1_estimator(s,a,theta,deg2rad(0.5),opts);
%
% Parsa Dini project - corrected research prototype, September 2026.

    if nargin < 5 || isempty(opts)
        opts = struct();
    end

    % ------------------------- defaults -------------------------
    opts = set_default(opts, 'reweight_iters', 2);
    opts = set_default(opts, 'enforce_projection_domain', true);
    opts = set_default(opts, 'max_mm_iter', 20);
    opts = set_default(opts, 'mm_tol', 1e-6);
    opts = set_default(opts, 'rank_tol', 1e-7);
    opts = set_default(opts, 'mu0_factor', 1e-3);
    opts = set_default(opts, 'mu_growth', 5);
    opts = set_default(opts, 'mu_max_factor', 1e3);
    opts = set_default(opts, 'max_mu_stages', 8);
    opts = set_default(opts, 'weight_floor_range', 1e-3);
    opts = set_default(opts, 'verbose', false);
    opts = set_default(opts, 'cvx_quiet', true);
    opts = set_default(opts, 'center', []);
    opts = set_default(opts, 'scale', []);

    % ----------------------- validation -------------------------
    if size(s,1) ~= 3 || size(a,1) ~= 3
        error('s and a must be 3-by-N matrices.');
    end
    n = size(s,2);
    if size(a,2) ~= n
        error('s and a must have the same number of columns.');
    end
    theta = theta(:).';
    if numel(theta) ~= n
        error('theta must contain one angle per observation.');
    end
    if any(~isfinite(theta)) || any(theta < 0) || any(theta > pi)
        error('theta must be finite and lie in [0, pi].');
    end

    if isscalar(sigma_theta)
        sigma_theta = repmat(sigma_theta, 1, n);
    else
        sigma_theta = sigma_theta(:).';
    end
    if numel(sigma_theta) ~= n || any(sigma_theta <= 0)
        error('sigma_theta must be positive scalar or 1-by-N.');
    end

    % Normalize each array axis.
    an = sqrt(sum(a.^2,1));
    if any(an <= eps)
        error('Each column of a must be nonzero.');
    end
    a = a ./ an;

    % ---------------- geometry normalization -------------------
    % Centering and scaling matter because the spectral rank penalty is not
    % invariant to translation/scale in homogeneous lifted coordinates.
    if isempty(opts.center)
        center = mean(s,2);
    else
        center = opts.center(:);
        if numel(center) ~= 3
            error('opts.center must be a 3-vector.');
        end
    end

    s_centered = s - center;
    if isempty(opts.scale)
        radii = sqrt(sum(s_centered.^2,1));
        scale = max(radii);
        if scale < 1e-9
            % Degenerate sensor cloud: fall back to absolute coordinate scale.
            scale = max(sqrt(sum(s.^2,1)));
        end
        if scale < 1e-9
            scale = 1;
        end
    else
        scale = opts.scale;
        if ~isscalar(scale) || ~isfinite(scale) || scale <= 0
            error('opts.scale must be a positive finite scalar.');
        end
    end
    sb = s_centered / scale;

    % Initial weighting: inverse angle variance only. A common scalar does
    % not change the geometric optimum, so normalize mean weight to one.
    w = 1 ./ (sigma_theta.^2);
    w = w / mean(w);

    history_all = struct('reweight_iter',{},'mu',{},'mm_iter',{}, ...
                         'objective',{},'rank_gap',{},'lambda2_over_1',{}, ...
                         'rel_Z_change',{},'cvx_status',{});

    Z = [];
    cvx_status_final = '';
    for rw = 1:max(1, opts.reweight_iters)
        [Z, hist_rw, cvx_status_final] = solve_rankpromoted_sdr(sb, a, theta, w, opts);
        for k = 1:numel(hist_rw)
            hist_rw(k).reweight_iter = rw;
            history_all(end+1) = hist_rw(k); %#ok<AGROW>
        end

        ubar = Z(1:3,end);

        if rw < opts.reweight_iters
            % Small-angle propagation:
            %   geometric residual g_i ~= -R_i * angle_error_i,
            % therefore Var(g_i) ~= R_i^2 sigma_i^2.
            Rbar = sqrt(sum((ubar - sb).^2,1));
            Rbar = max(Rbar, opts.weight_floor_range);
            w = 1 ./ (Rbar.^2 .* sigma_theta.^2);
            w = w / mean(w);
        end
    end

    [rank_gap, ratio21, evals] = rank_metrics(Z);
    ubar = Z(1:3,end);
    uhat = center + scale*ubar;

    % Principal-eigenvector reconstruction is reported for diagnostics only.
    % The moment estimate Z(1:3,end) is used as the primary output, because
    % it is directly tied to the homogeneous lifted constraints.
    [Ve,De] = eig((Z+Z')/2);
    [~,imax] = max(real(diag(De)));
    q = real(Ve(:,imax));
    if abs(q(end)) > 1e-12
        if q(end) < 0, q = -q; end
        q = q/q(end);
        u_eig = center + scale*q(1:3);
    else
        u_eig = nan(3,1);
    end

    result = struct();
    result.u = uhat;
    result.u_local = ubar;
    result.u_eig = u_eig;
    result.Z = Z;
    result.weights = w;
    result.rank_gap = rank_gap;
    result.lambda2_over_1 = ratio21;
    result.eigenvalues = evals;
    result.history = history_all;
    result.center = center;
    result.scale = scale;
    Cfinal = build_C(a, sb, theta, w);
    result.datafit = real(trace(Cfinal*Z));
    result.rank_penalty = real(trace(Z)-lambda_max_numeric(Z));
    result.cvx_status = cvx_status_final;
    result.model = 'theta is angle from +array axis: cos(theta)=a''(u-s)/||u-s||';
end

% =======================================================================
function [Z, history, cvx_status_out] = solve_rankpromoted_sdr(s, a, theta, w, opts)
    n = size(s,2);
    idx_u = 1:3;
    idx_r = 4:(3+n);
    idx_1 = n+4;
    dim = idx_1;

    C = build_C(a, s, theta, w);

    % Baseline SDR. No ad-hoc rho*trace(Z) term is included here; the rank
    % regularizer is introduced explicitly and separately below.
    if opts.cvx_quiet
        cvx_begin sdp quiet
    else
        cvx_begin sdp
    end
        variable Z(dim,dim) symmetric
        minimize( trace(C*Z) )
        subject to
            Z == semidefinite(dim);
            Z(idx_1,idx_1) == 1;
            for i = 1:n
                ai = a(:,i);
                si = s(:,i);
                Pi = eye(3) - ai*ai';

                Z(idx_r(i),idx_r(i)) == ...
                    trace(Pi*Z(idx_u,idx_u)) ...
                    - 2*si'*Pi*Z(idx_u,idx_1) ...
                    + si'*Pi*si;

                Z(idx_r(i),idx_1) >= 0;

                if opts.enforce_projection_domain
                    % Exact cone-distance branch condition:
                    % tau_i = d_i cos(theta_i) + r_i sin(theta_i) >= 0.
                    cos(theta(i)) * (ai'*(Z(idx_u,idx_1)-si)) ...
                    + sin(theta(i)) * Z(idx_r(i),idx_1) >= 0;
                end
            end
    cvx_end
    cvx_status_out = cvx_status;

    if ~status_acceptable(cvx_status)
        error('Initial SDR failed: CVX status = %s', cvx_status);
    end

    Cscale = max(norm(C,2), 1e-12);
    mu = opts.mu0_factor * Cscale;
    mu_max = opts.mu_max_factor * Cscale;

    history = struct('reweight_iter',{},'mu',{},'mm_iter',{}, ...
                     'objective',{},'rank_gap',{},'lambda2_over_1',{}, ...
                     'rel_Z_change',{},'cvx_status',{});

    stage = 0;
    while stage < opts.max_mu_stages
        stage = stage + 1;

        % FIX mu within this MM stage. This preserves the standard MM descent
        % interpretation for F_mu. Increase mu only after inner convergence.
        for mm = 1:opts.max_mm_iter
            Zold = Z;

            [V,D] = eig((Zold+Zold')/2);
            [~,imax] = max(real(diag(D)));
            q = real(V(:,imax));
            q = q / norm(q);
            Q = q*q';

            if opts.cvx_quiet
                cvx_begin sdp quiet
            else
                cvx_begin sdp
            end
                variable Z(dim,dim) symmetric
                minimize( trace(C*Z) + mu*(trace(Z) - trace(Q*Z)) )
                subject to
                    Z == semidefinite(dim);
                    Z(idx_1,idx_1) == 1;
                    for i = 1:n
                        ai = a(:,i);
                        si = s(:,i);
                        Pi = eye(3) - ai*ai';

                        Z(idx_r(i),idx_r(i)) == ...
                            trace(Pi*Z(idx_u,idx_u)) ...
                            - 2*si'*Pi*Z(idx_u,idx_1) ...
                            + si'*Pi*si;

                        Z(idx_r(i),idx_1) >= 0;

                        if opts.enforce_projection_domain
                            cos(theta(i)) * (ai'*(Z(idx_u,idx_1)-si)) ...
                            + sin(theta(i)) * Z(idx_r(i),idx_1) >= 0;
                        end
                    end
            cvx_end
            cvx_status_out = cvx_status;

            if ~status_acceptable(cvx_status)
                error('Rank-promoted SDP failed: CVX status = %s', cvx_status);
            end

            relZ = norm(Z-Zold,'fro') / max(1, norm(Zold,'fro'));
            [gap, ratio21, ~] = rank_metrics(Z);
            obj = real(trace(C*Z) + mu*(trace(Z)-lambda_max_numeric(Z)));

            h = struct();
            h.reweight_iter = NaN;
            h.mu = mu;
            h.mm_iter = mm;
            h.objective = obj;
            h.rank_gap = gap;
            h.lambda2_over_1 = ratio21;
            h.rel_Z_change = relZ;
            h.cvx_status = cvx_status;
            history(end+1) = h; %#ok<AGROW>

            if opts.verbose
                fprintf('mu=%9.3e  MM=%2d  rankGap=%9.3e  lambda2/lambda1=%9.3e  relZ=%9.3e\n', ...
                    mu, mm, gap, ratio21, relZ);
            end

            if relZ < opts.mm_tol
                break;
            end
        end

        [gap,~,~] = rank_metrics(Z);
        if gap <= opts.rank_tol || mu >= mu_max
            break;
        end

        mu = min(opts.mu_growth*mu, mu_max);
    end
end

% =======================================================================
function C = build_C(a, s, theta, w)
    n = size(s,2);
    idx_u = 1:3;
    idx_r = 4:(3+n);
    idx_1 = n+4;
    C = zeros(idx_1);

    for i = 1:n
        ci = zeros(idx_1,1);
        ci(idx_u) = -sin(theta(i))*a(:,i);
        ci(idx_r(i)) = cos(theta(i));
        ci(idx_1) = sin(theta(i))*(a(:,i)'*s(:,i));
        C = C + w(i)*(ci*ci');
    end
    C = (C+C')/2;
end

% =======================================================================
function [gap, ratio21, evals] = rank_metrics(Z)
    evals = sort(real(eig((Z+Z')/2)),'descend');
    evals(evals < 0 & evals > -1e-10) = 0;
    trZ = sum(evals);
    if trZ <= eps
        gap = NaN;
    else
        gap = max(0, trZ-evals(1))/trZ;
    end
    if numel(evals) >= 2 && evals(1) > eps
        ratio21 = max(0,evals(2))/evals(1);
    else
        ratio21 = 0;
    end
end

function lm = lambda_max_numeric(Z)
    lm = max(real(eig((Z+Z')/2)));
end

function tf = status_acceptable(status)
    tf = strcmpi(status,'Solved') || strcmpi(status,'Inaccurate/Solved');
end

function s = set_default(s, name, value)
    if ~isfield(s,name) || isempty(s.(name))
        s.(name) = value;
    end
end




% =====================================================================
% Sun et al. (IEEE TSP 2022) – exact implementation you provided
% =====================================================================
function sol = sun_et_al(s, a, theta, sigma)
%SUN_ET_AL  Sun, Ho et al., IEEE TSP 2022 (SDR + algebraic refinement)
    sol = [];
    if size(s,1) ~= 3, s = s.'; end
    if size(a,1) ~= 3, a = a.'; end
    theta = theta(:);
    [~, M] = size(s);
    if numel(theta) ~= M
        error('length(theta) must equal number of sensors');
    end
    a = a ./ vecnorm(a, 2, 1);
    sigma2 = max(sigma^2, 1e-16);
    Q = sigma2 * eye(M);
    try
        [uRfn, ~] = SA3DLocLA_SDR_core(theta, a, s, Q);
        if all(isfinite(uRfn))
            sol.u = uRfn(:);
        end
    catch
        sol = [];
    end
end

function [uRfn, uSDR] = SA3DLocLA_SDR_core(theta, gamma, S, Q)
    [N, M] = size(S);               % N=3, M=#sensors
    h = diag(gamma' * S);           % M×1
    G = [gamma', -diag(cos(theta))];% M×(3+M)
    B = eye(M);
    for itr = 1:2
        BBQreg = B*Q*B + 1e-12*eye(M);
        W = inv(BBQreg);
        A = [G'*W*G, -G'*W*h;
            -h'*W*G,  h'*W*h];
        cvx_begin quiet
            cvx_solver sdpt3
            variable P(M+N,M+N) symmetric
            variable p(M+N,1)
            SM = [P, p; p', 1];
            minimize( trace(SM*A) )
            subject to
                for i = 1:M
                    P(i+N,i+N) == ...
                        trace(P(1:N,1:N)) - 2*S(:,i)'*p(1:N) + S(:,i)'*S(:,i);
                    p(i+N) >= 0;
                end
                for i = 1:M-1
                    for j = i+1:M
                        P(i+N,j+N) >= abs( ...
                            trace(P(1:N,1:N)) ...
                            - S(:,i)'*p(1:N) - S(:,j)'*p(1:N) ...
                            + S(:,i)'*S(:,j) );
                    end
                end
                SM == semidefinite(M+N+1);
        cvx_end
        if ~(strcmp(cvx_status,'Solved') || strcmp(cvx_status,'Inaccurate/Solved'))
            uSDR = nan(3,1);
            uRfn = nan(3,1);
            return;
        end
        u2   = sqrt(max(diag(SM(1:N,1:N)), 0));
        uSDR = func_findu(theta, gamma, S, Q, u2, M);
        r = sqrt(sum((S - uSDR).^2, 1)).';
        B = -diag(r .* sin(theta));
    end
    % algebraic refinement
    uRfn = uSDR;
    for iter = 1:2
        r = sqrt(sum((S - uRfn).^2, 1)).';
        g = h - gamma'*uRfn + diag(cos(theta))*r;
        Ht = zeros(N+M, N);
        Ht(1:N,1:N) = eye(N);
        for i = 1:M
            d = uRfn - S(:,i);
            Ht(N+i,:) = d' / max(norm(d), eps);
        end
        H2  = G * Ht;
        du2 = -(H2'*W*H2 + 1e-12*eye(N)) \ (H2'*W*g);
        uRfn = uRfn - du2;
        r = sqrt(sum((S - uSDR).^2, 1)).';
        B = -diag(r .* sin(theta));
        BBQreg = B*Q*B + 1e-12*eye(M);
        W = inv(BBQreg);
    end
end

function uu = func_findu(theta, gamma, S, Q, pos, M)
    sgn = [ 1  1  1;
            1  1 -1;
            1 -1  1;
            1 -1 -1;
           -1  1  1;
           -1  1 -1;
           -1 -1  1;
           -1 -1 -1].';
    J = inf(8,1);
    for j = 1:8
        u = sgn(:,j) .* pos;
        est = zeros(M,1);
        for i = 1:M
            v = u - S(:,i);
            nv = norm(v);
            if nv < eps, continue; end
            c = gamma(:,i)'*v / nv;
            c = max(min(c,1),-1);
            est(i) = acos(c);
        end
        J(j) = (theta - est)' / Q * (theta - est);
    end
    [~,ind] = min(J);
    uu = sgn(:,ind) .* pos;
end


% =====================================================================
% O-SDR (Ordinary SDR) 
% =====================================================================
function result = conical_sdr_baseline(s, a, theta, weights, opts)
%CONICAL_SDR_BASELINE Corrected ordinary SDR for conical localization.
%
% theta is the angle from the oriented +array axis:
%   cos(theta_i) = a_i'*(u-s_i)/||u-s_i||, theta_i in [0,pi].
%
% weights is optional. Use all ones for unweighted geometric least squares.
% This routine does NOT add a trace regularizer and does NOT promote rank one.
%
% Requires CVX for MATLAB.

    if nargin < 4 || isempty(weights), weights = []; end
    if nargin < 5 || isempty(opts), opts = struct(); end
    if ~isfield(opts,'enforce_projection_domain'), opts.enforce_projection_domain = true; end
    if ~isfield(opts,'cvx_quiet'), opts.cvx_quiet = true; end
    if ~isfield(opts,'center'), opts.center = []; end
    if ~isfield(opts,'scale'), opts.scale = []; end

    if size(s,1)~=3 || size(a,1)~=3 || size(s,2)~=size(a,2)
        error('s and a must be 3-by-N with the same N.');
    end
    n = size(s,2);
    theta = theta(:).';
    if numel(theta)~=n || any(theta<0) || any(theta>pi)
        error('theta must be 1-by-N in [0,pi].');
    end
    if isempty(weights), weights = ones(1,n); end
    weights = weights(:).';
    if numel(weights)~=n || any(weights<=0)
        error('weights must be positive scalar/vector with N entries.');
    end
    weights = weights/mean(weights);

    an = sqrt(sum(a.^2,1));
    if any(an<=eps), error('Array axis columns must be nonzero.'); end
    a = a./an;

    if isempty(opts.center)
        center = mean(s,2);
    else
        center = opts.center(:);
    end
    sc = s-center;
    if isempty(opts.scale)
        scale = max(sqrt(sum(sc.^2,1)));
        if scale < 1e-9, scale = max(sqrt(sum(s.^2,1))); end
        if scale < 1e-9, scale = 1; end
    else
        scale = opts.scale;
    end
    sb = sc/scale;

    idx_u=1:3; idx_r=4:(3+n); idx_1=n+4; dim=idx_1;
    C=zeros(dim);
    for i=1:n
        ci=zeros(dim,1);
        ci(idx_u)=-sin(theta(i))*a(:,i);
        ci(idx_r(i))=cos(theta(i));
        ci(idx_1)=sin(theta(i))*(a(:,i)'*sb(:,i));
        C=C+weights(i)*(ci*ci');
    end
    C=(C+C')/2;

    if opts.cvx_quiet
        cvx_begin sdp quiet
    else
        cvx_begin sdp
    end
        variable Z(dim,dim) symmetric
        minimize(trace(C*Z))
        subject to
            Z == semidefinite(dim);
            Z(idx_1,idx_1)==1;
            for i=1:n
                ai=a(:,i); si=sb(:,i); Pi=eye(3)-ai*ai';
                Z(idx_r(i),idx_r(i)) == ...
                    trace(Pi*Z(idx_u,idx_u)) ...
                    -2*si'*Pi*Z(idx_u,idx_1) + si'*Pi*si;
                Z(idx_r(i),idx_1)>=0;
                if opts.enforce_projection_domain
                    cos(theta(i))*(ai'*(Z(idx_u,idx_1)-si)) ...
                    + sin(theta(i))*Z(idx_r(i),idx_1) >= 0;
                end
            end
    cvx_end

    ubar=Z(idx_u,idx_1);
    u=center+scale*ubar;
    ev=sort(real(eig((Z+Z')/2)),'descend');
    gap=max(0,sum(ev)-ev(1))/max(sum(ev),eps);
    if numel(ev)>1 && ev(1)>eps, r21=max(0,ev(2))/ev(1); else, r21=0; end

    datafit=real(trace(C*Z));
    result=struct('u',u,'u_local',ubar,'Z',Z,'rank_gap',gap, ...
        'lambda2_over_1',r21,'eigenvalues',ev,'center',center, ...
        'scale',scale,'cvx_status',cvx_status,'weights',weights, ...
        'datafit',datafit);
end
