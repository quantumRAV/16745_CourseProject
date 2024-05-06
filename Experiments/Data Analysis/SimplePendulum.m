clear all
close all
%% Simple Pendulum. 
% Discretize continuous-time dynamics using RK3 and roll out the states
% Note: theta = 0 corresponds to the pendulum hanging straight down. theta
% = pi/2 correspends to the pendulum pointing to the right

m = 0.1; %kg
g = 9.81; %m/s^2
l = 0.3; %m

dt = 0.05; %time step in seconds
N = 1000; %total number of steps to simulate
t = [1:N]*dt;

x0 = [pi/2;0]; %zero initial conditions. 1st element is the position, 2nd is the velocity
u = zeros(1,N); %No applied torque

%simulate using RK4
x_RK4 = zeros(2,N);
x_RK4(:,1) = x0; %initial condition

pen_dyn = @(x,u) pendulum_dynamics_continuous(x,u,m,g,l);

for ii=1:N-1
    x_RK4(:,ii+1) = RK4(pen_dyn, x_RK4(:,ii), u(ii), dt);  
end

figure()
subplot(2,1,1)
hold on
plot(t,x_RK4(1,:),'r--', DisplayName = "Position");
xlabel('time (s)')
ylabel('Angular position $\theta$ (rad)', 'Interpreter','latex')

subplot(2,1,2)
hold on
plot(t,x_RK4(2,:),'r--', DisplayName = "Angular Velocity");
xlabel('time (s)')
ylabel('Angular velocity, $\dot{\theta}$ (rad/s))', 'Interpreter','latex')

%% fit linear model to pendulum data using N4sid, and compare the A and B matrices
figure(2)
nx = 1:10; %model order
sys = n4sid(u.',x_RK4(1,:).', nx, 'Ts', dt);

compare(u.',x_RK4(1,:).',sys);
hold on
plot(t,x_RK4(1,:),'r--', DisplayName = "Angular position from RK4");

%Display the A, B and C matrices. Without forcing (i.e. u = 0), we would
%expect the B matrix to be close to 0

disp("A:")
disp(sys.A)

disp("B:")
disp(sys.B)

disp("C:")
disp(sys.C)

% Rollout the data from n4sid to put it into a matrix form for plotting
n4_sim_results = zeros(1, length(u));

% init position of y is the first position of x_0
y_k = x_RK4(1,1); 
for i=1:length(u)
    x_k = pinv(sys.C)*y_k;
    x_k2 = sys.A*x_k + sys.B*u(i);
    
    % update y_k with this updated timestep for next loop and add it to
    % results
    y_k = sys.C*x_k2; % y_k effectively now y_k+1 for next loop
    n4_sim_results(1,i) = y_k;
end

figure(5)
legend
hold on
xlabel('Time (s)')
ylabel('Angular position, $\theta$ (rad)', 'Interpreter','latex')
plot(t, n4_sim_results, DisplayName="N4sid Matrix Rollout")
plot(t, x_RK4(1,:), '--', DisplayName="RK4 Pendulum Dynamics")

%% Fit using homebrew dynamics regression
nx = 1; % theta
nu = 1; % torque
nm = length(x_RK4(1,:)); % number of measurements = length of first index

X = zeros(nx, nm);
U = zeros(nu, nm);

X = x_RK4(1,:);
U(1,:) = u;

% shifting to massage the data such that our collected data is in the form:
% x_k2 = f(x_k1, u_k1)
U_k1 = U(:, 1:end-1);
X_k1 = X(:, 1:end-1);
X_k2 = X(:, 2:end);

% Naive A and B, i.e. without any time delays
%[naive_A_tilde, naive_B_tilde] = Dynamics_Mat_Reg(gen_X_k1, gen_U_k1, gen_X_k2, 1, 1);

% Fitting A and B w/o kron approach
naive_A = zeros(nx, nx); % standard A is nx by nx
naive_B = zeros(nx, nu); % standard B is nx by nu

% stack controls and state for regression
state_ctrl = [U_k1; X_k1];

% linear regression to solve for dynamics matrices, will be in form: 
% dynamics_matrix = [naive_B naive_A]
dynamics_matrix = linsolve(state_ctrl', X_k2')';

% sizes should match original declaration
naive_B = dynamics_matrix(:, 1:nu);
naive_A = dynamics_matrix(:, nu+1:end); 
% (also these should match the original answer from vec trick, A_tilde,
% B_tilde)

%% Roll the naive stuff forward to compare (presumably this will perform terribly)
init_cond = X_k1(:,1);
naive_X_sim = Delay_no_kron_Forward_Rollout(dynamics_matrix, init_cond,...
    nx, U_k1, 0);

% shift the time over a step so it maps to our x_k+1 output
sim_t = t(:,2:end);

% (the plot does indeed look like nothing)
%plot(sim_t, naive_X_sim(1,:),'--o', DisplayName = "Angular Position from Naive Linear Regression");

%% Test some time delay embedding

% some sample delay quantities
test_delays = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50];

% cell matrices to store x_k, x_k+1, u_k, [B A_i] for all tested delays
% first row = x_k, second row = x_k+1
% third row = rollout simulation, fourth row = time (for plotting)
X_delays = cell(5, length(test_delays));
U_delays = cell(1, length(test_delays));
dyn_mat_delays = cell(1, length(test_delays));

errors = zeros(1, length(test_delays));

% generate respective time delay embedded matrices
for i=1:length(test_delays)
    % x_k gets the time delays embedded into it
    X_delays{1, i} = Time_Delay_Embed(X_k1, test_delays(i));
    
    % x_k2, u_k, & time just get trimmed so their length/entries stay in
    % alignment
    % (ie if x_k starts at 2, then we need u_k to also start at 2)
    U_delays{1, i} = U_k1(:, test_delays(i)+1:end);
    X_delays{2, i} = X_k2(:, test_delays(i)+1:end);
    X_delays{4, i} = sim_t(:, test_delays(i)+1:end); % time row
    
    % stack controls and state for regression
    state_ctrl = [U_delays{1, i}; X_delays{1, i}];

    % regression step
    dyn_mat_delays{1, i} = X_delays{2,i}*pinv(state_ctrl);

    % rolling the system out
    init_cond = X_delays{1, i}(:,1);
    X_delays{3, i} = Delay_no_kron_Forward_Rollout(dyn_mat_delays{1, i},...
        init_cond, nx, U_delays{1,i}, test_delays(i));

    % calc RMSE between rollout and actual
    errors(i) = rmse(X_delays{3,i}, X_delays{2,i});
end

% ex) test_delay(1) => 1 delay
%plot(X_delays{4, 1}, X_delays{3,1},'--+', DisplayName = "1 Delay");

figure(3)
legend
hold on
xlabel('Time (s)')
ylabel('Angular position, $\theta$ (rad)', 'Interpreter','latex')
plot(t,x_RK4(1,:),'r-', DisplayName = "Position")
plot(X_delays{4, 1}, X_delays{3,1},'b-', DisplayName = "1 Delay");
plot(X_delays{4, 3}, X_delays{3,3},'g--', DisplayName = "3 Delay");

figure(4)
hold on
xlabel('Number of Time Delays')
ylabel('RMSE with Actual Data')
plot(test_delays, errors, '-o')

%figure(2)
%plot(X_delays{4, 3}, X_delays{3,3},'--o', DisplayName = "3 Delay");
%plot(X_delays{4, 2}, X_delays{3,2},'--o', DisplayName = "2 Delay")

%% Looking at SVD side
svd_mats = cell(3, length(test_delays));
for i=1:length(test_delays)
    % [U, S, V] = svd(A, "econ") yields A = U*S*V'
    [svd_mats{1, i}, svd_mats{2, i}, svd_mats{3, i}] = svd(X_delays{1, i}, "econ", "vector");
end

figure(8)
svd_eigen_plot = tiledlayout(4, 1, "TileSpacing","tight");

% overall label options
title(svd_eigen_plot, 'Eigenvalue Relative Correlation Across Delays (Pendulum Simulation)')
xlabel(svd_eigen_plot, 'SVD Eigenvalue ($\sigma_i$)', 'Interpreter', 'latex')
ylabel(svd_eigen_plot, '$\sigma_i$ / $\sum\sigma$', 'Interpreter','latex')

% Tile 1
nexttile
plot(svd_mats{2,1}/sum(svd_mats{2,1}), '-o')
title("1 Delay SVD Eigenvalue Correlation")
xlim([0, 10])

% Tile 2
nexttile
plot(svd_mats{2,5}/sum(svd_mats{2,5}), '-o')
title("5 Delay SVD Eigenvalue Correlation")
xlim([0, 10])

% Tile 3
nexttile
plot(svd_mats{2,10}/sum(svd_mats{2,10}), '-o')
title("10 Delay SVD Eigenvalue Correlation")
xlim([0, 10])

% Tile 4
nexttile
plot(svd_mats{2,11}/sum(svd_mats{2,11}), '-o')
title("20 Delay SVD Eigenvalue Correlation")
xlim([0, 10])

%% Dynamics and integration functions
function x_dot = pendulum_dynamics_continuous(x,u,m,g,l)
    x_dot=zeros(2,1);
    x_dot(1) = x(2); %velocity kinematics
    x_dot(2) = -(g/l)*sin(x(1)) + u./(m*(l^2)); %continuous pendulum dynamics
end

function xk_1 = RK4(f, x, u, dt)

    k1 = f(x,u);
    k2 = f(x + 0.5*dt*k1,u);
    k3 = f(x + 0.5*dt*k2,u);
    k4 = f(x + dt*k3,u);

    xk_1 = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4); %next state

end

