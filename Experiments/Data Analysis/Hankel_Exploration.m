%% Looking into Time-Delay Embedding for Modeling Grasper
%
% Random step testing has been performed as inspo'd by Haggerty Sci Rob
% Paper. Based on step inputs to the McKibben actuator, pressure at the
% jaws was monitored as a proxy for the force exerted by the grasper
% system.
%
% Looking to leverage Time-Delay Embedding as a means of fitting a
% linear dynamics
% model to this soft, non-linear system, such that our optimal controls
% approaches can be leveraged from class such that the grasper may be
% commanded to impart a particular force trajectory.
%
% N. Zimmerer & R. Sukhnandan

%clc % clear cmd window
close all % closes all figs
clear  % clear workspace

disp('hello')

%% Import Test Data
% Data gathered with random, varied step inputs, to grasp a 40 mm diameter
% cylinder. Estimated input pressure to contact -> 7 psi
cd('Koopman Data/')

% reads data into an array of doubles (ie any strings will be NaN)
test_data = readmatrix('Koopman_Testing_06_04_2024_10_04_48.csv');
test_data2 = readmatrix('Koopman_Testing_30_04_2024_17_07_00.csv');
cd .. % back out to the .m level workspace

% commanded McKibben pressure
control_input = test_data(2:end, 17); % skip the first row on account of label
control_input2 = test_data2(2:end, 17);

% read in jaw pressures
jaw1_pressure = test_data(2:end, 14);
jaw2_pressure = test_data(2:end, 15);
jaw3_pressure = test_data(2:end, 16);

jaw1_pressure2 = test_data2(2:end, 14);
jaw2_pressure2 = test_data2(2:end, 15);
jaw3_pressure2 = test_data2(2:end, 16);


% test timing
time = test_data(2:end, 6);
time = time - time(1); % for plotting set t0 = 0s
time = time'; % align time as a row vector

time2 = test_data2(2:end, 6);
time2 = time2-time2(1);
time2 = time2';

%% Formatting Data
% many options for how to shape the data here, let's start with the
% standard big boi of stacking the states where each col is a timestep
nx = 3; % using all three jaw states
nu = 1; % using a single commanded state
nm = length(jaw1_pressure); % number of measurements = length of first index
nm2 = length(jaw1_pressure2); % second experiment

gen_X = zeros(nx, nm);
gen_U = zeros(nu, nm);

exp2_X = zeros(nx, nm2);
exp2_U = zeros(nu, nm2);

% fill in the state and control matrices
for i=1:length(jaw1_pressure)
    gen_X(:, i) = [jaw1_pressure(i); jaw2_pressure(i); jaw3_pressure(i)];
    gen_U(:, i) = [control_input(i)];
end

for i=1:length(jaw1_pressure2)
    exp2_X(:,i) = [jaw1_pressure2(i); jaw2_pressure2(i); jaw3_pressure2(i)];
    exp2_U(:,i) = [control_input2(i)];
end

% technically this sizing is a bit off from what we normally use where
% x_k+1 = Ax_k + Bu_k. Based on measurements, easiest to lop off the final
% column of u for u_k, same for x_k, then lop off first col to make x_k+1
gen_U_k1 = gen_U(:, 1:end-1);
gen_X_k1 = gen_X(:, 1:end-1);
gen_X_k2 = gen_X(:, 2:end);

exp2_U_k1 = exp2_U(:, 1:end-1);
exp2_X_k1 = exp2_X(:, 1:end-1);
exp2_X_k2 = exp2_X(:, 2:end);

% maintain trimmed time rows to follow x_k and x_k2
gen_X_k1t = time(:, 1:end-1);
gen_X_k2t = time(:, 2:end);

exp2_X_k1t = time2(:, 1:end-1);
exp2_X_k2t = time2(:, 2:end);

%% No Time Delay Embedding or Data Parsing
% First check of our process/approach and functions. Plugging all of our
% data naively into the Dynamics_Mat_Reg function should result in a linear
% least squares set of A and B matrices fit to the data
% [naive_A, naive_B] = Dynamics_Mat_Reg(gen_X_k1, gen_U_k1, gen_X_k2, 3, 1);

% Fitting A and B w/o kron approach
naive_A = zeros(nx, nx); % standard A is nx by nx
naive_B = zeros(nx, nu); % standard B is nx by nu

naive_A2 = zeros(nx, nx);
naive_B2 = zeros(nx, nu);

% stack controls and state for regression
state_ctrl = [gen_U_k1; gen_X_k1];
state_ctrl2 = [exp2_U_k1; exp2_X_k2];

% linear regression to solve for dynamics matrices, will be in form: 
% dynamics_matrix = [naive_B naive_A]
dynamics_matrix = linsolve(state_ctrl', gen_X_k2')';
dynamics_matrix2 = linsolve(state_ctrl2', exp2_X_k2')';

% sizes should match original declaration
naive_B = dynamics_matrix(:, 1:nu);
naive_A = dynamics_matrix(:, nu+1:end);
% (also these should match the original answer from vec trick, A_tilde,
% B_tilde)

naive_B2 = dynamics_matrix2(:, 1:nu);
naive_A2 = dynamics_matrix2(:, nu+1:end);


%% Non Partitioned Data w/ Time Delay Embedding (avoid kron and vec)
% Time_Delay_Embed(matrix_to_delay, number of delays)
% parameters to tweak -> number of delays used

delays = 1:20; % some delays to try

% cell matrices to store x_k, x_k+1, u_k, [B A_i] for all tested delays
% first row = x_k, second row = x_k+1
% third row = rollout simulation, fourth row = time (for plotting)
gen_X_delays = cell(5, length(delays));
gen_U_delays = cell(1, length(delays)); 
gen_dyn_matrices_delays = cell(1, length(delays));

errors = zeros(1, length(delays));

exp2_X_delays = cell(5, length(delays));
exp2_U_delays = cell(1, length(delays)); 
exp2_dyn_matrices_delays = cell(1, length(delays));

errors2 = zeros(1, length(delays));

% generate time delay embedded matrices for varying delays
for i=1:length(delays)
    % x_k gets time delays baked into it
    gen_X_delays{1, i} = Time_Delay_Embed(gen_X_k1, delays(i));
    exp2_X_delays{1, i} = Time_Delay_Embed(exp2_X_k1, delays(i));
    
    % x_k2, u_k, & time just get trimmed so their length/entries stay in
    % alignment
    % (ie if x_k starts at 2, then we need u_k to also start at 2)
    gen_U_delays{1, i} = gen_U_k1(:, delays(i)+1:end);
    gen_X_delays{2, i} = gen_X_k2(:, delays(i)+1:end);
    gen_X_delays{4, i} = gen_X_k2t(:,delays(i)+1:end); % time row

    exp2_U_delays{1, i} = exp2_U_k1(:, delays(i)+1:end);
    exp2_X_delays{2, i} = exp2_X_k2(:, delays(i)+1:end);
    exp2_X_delays{4, i} = exp2_X_k2t(:,delays(i)+1:end);    
    
    % stack controls and state for regression
    state_ctrl = [gen_U_delays{1, i}; gen_X_delays{1, i}];
    state_ctrl2 = [exp2_U_delays{1, i}; exp2_X_delays{1, i}];
    
    % linear regression step
    gen_dyn_matrices_delays{1, i} = linsolve(state_ctrl',gen_X_delays{2,i}')';
    exp2_dyn_matrices_delays{1, i} = linsolve(state_ctrl2',exp2_X_delays{2,i}')';

    % rolling the system out
    init_cond = gen_X_delays{1, i}(:,1);
    gen_X_delays{3, i} = Delay_no_kron_Forward_Rollout(gen_dyn_matrices_delays{1, i},...
        init_cond, nx, gen_U_delays{1,i}, delays(i));

    init_cond = exp2_X_delays{1,i}(:,1);
    exp2_X_delays{3, i}=Delay_no_kron_Forward_Rollout(exp2_dyn_matrices_delays{1,i},...
        init_cond, nx, exp2_U_delays{1,i}, delays(i));

    % calc RMSE between rollout and actual
    errors(i) = rmse(gen_X_delays{3,i}, gen_X_delays{2,i}, "all");   
    errors2(i) = rmse(exp2_X_delays{3,i}, exp2_X_delays{2,i}, "all");
end

%% Some initial plots

figure(1)
hold on
title('RMS for Time Delays Across General Dataset')
xlabel('Number of Time Delays')
ylabel('RMSE with Actual Data')
plot(delays, errors, '-o')

figure(2)
plot(gen_X_k2t, gen_X_k2(1,:),'r--',DisplayName="Exp Data")
hold on
legend
xlabel('Time (s)')
ylabel('Jaw 1 Pressure (psi)')
ax = gca; % get current axis (gca)
ax.FontSize = 22;
plot(gen_X_delays{4,1}, gen_X_delays{3,1}(1,:), DisplayName="1 Delay")
plot(gen_X_delays{4,5}, gen_X_delays{3,5}(1,:), DisplayName="5 Delays")
plot(gen_X_delays{4,8}, gen_X_delays{3,8}(1,:), DisplayName="8 Delays")
plot(gen_X_delays{4,20}, gen_X_delays{3,20}(1,:), DisplayName="20 Delays")

%% Grasp V Free Grouping
% Work with the dataset set from second batch of testing that was predominantly
% maintaining contact 
figure(3)
hold on
title('RMS for Time Delays Across Contact Dataset')
xlabel('Number of Time Delays')
ylabel('RMSE with Actual Data')
plot(delays, errors2, '-o')

figure(4)
plot(exp2_X_k2t, exp2_X_k2(1,:),'r--',DisplayName="Exp Data")
hold on
legend
xlabel('Time (s)')
ylabel('Jaw 1 Pressure (psi)')
ax = gca; % get current axis (gca)
ax.FontSize = 22;
plot(exp2_X_delays{4,1}, exp2_X_delays{3,1}(1,:), DisplayName="1 Delay")
plot(exp2_X_delays{4,5}, exp2_X_delays{3,5}(1,:), DisplayName="5 Delays")
plot(exp2_X_delays{4,8}, exp2_X_delays{3,8}(1,:), DisplayName="8 Delays")
plot(exp2_X_delays{4,20}, exp2_X_delays{3,20}(1,:), DisplayName="20 Delays")

% visualize some of the general data from 40 cm cylinder experiment
figure(5)
hold on
legend
ax3 = gca;
ax3.FontSize = 22;
xlabel('Time (s)');
title('40cm Cylinder Test Data (4/30/24)')
yyaxis left;
ylabel('Jaw 1 Pressure (psi)')
ax3L = gca;
ax3L.YColor = 'black';
plot(exp2_X_k2t, exp2_X_k2(1,:),'r--',DisplayName="Exp Data")
yyaxis right;
ylabel('Input Pressure (psi)')
ax3R = gca;
ax3R.YColor = 'black';
plot(exp2_X_k1t, exp2_U_k1(1,:), 'b-', DisplayName="Control Input")

% Pull apart data to isolate times when the grasper is in contact
% with the test cylinder (40 mm diam, in contact ~ u > 7 psi)

% visualize some of the general data from 40 cm cylinder experiment
figure(6)
hold on
legend
ax3 = gca;
ax3.FontSize = 22;
xlabel('Time (s)');
title('40cm Cylinder Test Data (4/6/24)')
yyaxis left;
ylabel('Jaw 1 Pressure (psi)')
ax3L = gca;
ax3L.YColor = 'black';
plot(gen_X_k2t, gen_X_k2(1,:),'r--',DisplayName="Exp Data")
yyaxis right;
ylabel('Input Pressure (psi)')
ax3R = gca;
ax3R.YColor = 'black';
plot(gen_X_k1t, gen_U_k1(1,:), 'b-', DisplayName="Control Input")

% based on this sset of data, the responsiveness from Jaw 1 at least
% indicates that for most of the test inputs > 7.05 psi, we appear to have
% some responsive contact. Let's pull out the indices where this condition
% occurs
contact_indices_40cm = find(gen_U_k1 > 7.05);
contact_U_k = gen_U_k1(:,contact_indices_40cm);
% need to be smarter here on how to build up the x_ks so the time embedding works properly
contact_X_k1 = gen_X_k1(:,contact_indices_40cm); 
contact_X_k2 = gen_X_k2(:,contact_indices_40cm);
contact_X_k2t = gen_X_k2t(:,contact_indices_40cm);

figure(7)
%yyaxis left;
hold on
plot(contact_X_k2t, contact_X_k2(1,:), 'black--', DisplayName="Contact Data")
plot(gen_X_k2t, gen_X_k2(1,:),'r--',DisplayName="Exp Data")

%% Cross Validation Partitioning
% TODO: Judiciously partition data into training vs test sets as well
% (currently have an 80/20 split in mind). Need to take care not to break
% apart contiguous timing. I'm picturing having collections of vector
% datasets that get randomly assigned which then would become Hankel
% matrices for training if in the training set.
%
% This is in contrast to just randomly plucking 80% of points from a set. I
% think this is an important distinction given reliance on time embedding
% continuity.

%% SVD Matrices for Each Time Embedded Sys
% row 1 = U, row 2 = S, row 3 = V
svd_mats = cell(3, length(delays));
for i=1:length(delays)
    % [U, S, V] = svd(A, "econ") yields A = U*S*V'
    [svd_mats{1, i}, svd_mats{2, i}, svd_mats{3, i}] = svd(exp2_X_delays{1, i}, "econ", "vector");
end

%% Build up Hankel Matrices - Obsolete (See Time Delay Embed)
% Zach's explanation aligns with Haggerty paper that suggests an altered
% form of this data structure for us to use
%
% Use the construct hankel function to build up X_lift and X_lift+
% matrices for different selections of datasets.
%
% Additional parameters to tweak -> number of delays used because there's a
% tradeoff in Hankel matrix sizing based on the quantity of delays used
%
% Construct_Hankel(vector, nx, np, nm)
% nx = number of states
% np = number of delays
% nm = number of measurements
%
% trying it with general, giant state matrix
%nd = 4; % number of delays
%gen_X_H1 = Construct_Hankel(gen_X_k1, gen_nx, nd, length(gen_X_k1(1,:))); % H
%gen_X_H2 = Construct_Hankel(gen_X_k2, gen_nx, nd, length(gen_X_k2(1,:))); % H+

%% Dynamic V Stable Grouping
% TODO: pull apart data to isolate when behavior is in a dynamic vs stable
% state

%% Calculation of a Static Gain
% Akin to Haggerty, with static data separated, calculate a static gain to
% bolster the control

%% Random notes
% apparently the n4sid toolbox may have some usefulness to us here
% Ref: https://www.mathworks.com/help/ident/ref/n4sid.html