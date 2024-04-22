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
clear % clear workspace

disp('hello')

%% Import Test Data
% Data gathered with random, varied step inputs, to grasp a 40 mm diameter
% cylinder. Estimated input pressure to contact -> 7 psi
cd('Koopman Data/')

% reads data into an array of doubles (ie any strings will be NaN)
test_data = readmatrix('Koopman_Testing_06_04_2024_10_04_48.csv');
cd .. % back out to the .m level workspace

% commanded McKibben pressure
control_input = test_data(2:end, 17); % skip the first row on account of label

% TODO: read in commanded x and y positions (if/when we get there)

% read in jaw pressures
jaw1_pressure = test_data(2:end, 14);
jaw2_pressure = test_data(2:end, 15);
jaw3_pressure = test_data(2:end, 16);

% TODO: read in x and y positions (for when/if we incorporate this)

%% Formatting Data
% many options for how to shape the data here, let's start with the
% standard big boi of stacking the states where each col is a timestep
gen_nx = 3; % using all three jaw states
gen_nu = 1; % using a single commanded state
gen_nm = length(jaw1_pressure); % number of measurements = length of first index

gen_X = zeros(gen_nx, gen_nm);
gen_U = zeros(gen_nu, gen_nm);

% fill in the state and control matrices
for i=1:length(jaw1_pressure)
    gen_X(:, i) = [jaw1_pressure(i); jaw2_pressure(i); jaw3_pressure(i)];
    gen_U(:, i) = [control_input(i)];
end

% technically this sizing is a bit off from what we normally use where
% x_k+1 = Ax_k + Bu_k. Based on measurements, easiest to lop off the final
% column of u for u_k, same for x_k, then lop off first col to make x_k+1
gen_U_k1 = gen_U(:, 1:end-1);
gen_X_k1 = gen_X(:, 1:end-1);
gen_X_k2 = gen_X(:, 2:end);

%% No Time Delay Embedding or Data Parsing
% First check of our process/approach and functions. Plugging all of our
% data naively into the Dynamics_Mat_Reg function should result in a linear
% least squares set of A and B matrices fit to the data
[naive_A, naive_B] = Dynamics_Mat_Reg(gen_X_k1, gen_U_k1, gen_X_k2, 3, 1);

% right now this returns zero matrices for A and B which is obviously not
% super helpful.. I can't help but wonder if I have the wrong
% interpretation of how to solve these systems (ie is doing out the sum and
% then my application of the least squares setup incorrect?)

%% Non-Partitioned Data w/ Time Delay Embedding
% Generating a variety of time embeddings for the whole spectrum / range of
% data to generate some A and B dynamics matrices fit to the data. Results
% from this should end up mirroring what matlab's n4sid currently gives us
% (at least we think..)

% Time_Delay_Embed(matrix_to_delay, number of delays)
% parameters to tweak -> number of delays used

delays = 1:10; % some delays to try

% need to store our different time delay embeddings
gen_X_k1_delays = cell(1, length(delays));
gen_X_k2_delays = cell(1, length(delays));
gen_U_k1_delays = cell(1, length(delays));

% somewhere to store the dynamics matrices that result from this time delay
% embedding
gen_dyn_matrices_delays = cell(1, length(delays));

% generate time delay embedded matrices for varying quantities of time
% delays
for i=1:length(delays)
    gen_X_k1_delays{1, i} = Time_Delay_Embed(gen_X_k1, delays(i));
    gen_X_k2_delays{1, i} = Time_Delay_Embed(gen_X_k2, delays(i));
    gen_U_k1_delays{1, i} = Time_Delay_Embed(gen_U_k1, delays(i));

    % fit dynamics matrices to these different sets of delays
    gen_dyn_matrices_delays{1, i} = Dynamics_Mat_Reg(gen_X_k1_delays{1, i},...
        gen_U_k1_delays{1, i}, gen_X_k2_delays{1, i},...
        gen_nx*(delays(i)+1), gen_nu*(delays(i)+1));
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

%% Grasp V Free Grouping
% TODO: pull apart data to isolate times when the grasper is in contact
% with the test cylinder (40 mm diam, in contact ~ u > 7 psi)

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

%% Dynamic V Stable Grouping
% TODO: pull apart data to isolate when behavior is in a dynamic vs stable
% state

%% Calculate A and B Matrices
% TODO: From my understanding of the Haggerty paper, here's where a pseudo
% inverse is calculated as a solution to the optimization problem finding A
% and B.
%
% Additional knob to tune here is perhaps with the calculation: 
% How many terms do we keep in the pseudo inverse to discourage overfitting
% Or like L1 regularization was used to encourage sparsity + discourage
% overfitting

% I'm a little unsure about sizing here (ie won't the A and B matrices in
% this case be quite huge (because our lifted state is no longer just a
% column vector?) More importantly, I'm a little unsure how a real time
% roll out will work -> given the sizing, do we have to acquire some sort
% of data before the system can do anything? maybe the answer is to couch
% it at the start with steady state initial conditions...?

%% Rollout A and B Matrix Calculation on Test Set
% TODO: perform some validating rollout of the data forwards to see how the
% A and B matrices do at recreating dynamic portions of the set aside test
% sets (can build a X_test_lift and then compare it to the actual
% X_test_lift+)?

%% Calculation of a Static Gain
% Akin to Haggerty, with static data separated, calculate a static gain to
% bolster the control

%% Random notes
% apparently the n4sid toolbox may have some usefulness to us here
% Ref: https://www.mathworks.com/help/ident/ref/n4sid.html