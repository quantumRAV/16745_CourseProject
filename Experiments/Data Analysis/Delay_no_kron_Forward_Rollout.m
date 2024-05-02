%% Use A_tilde and B_tilde matrices to calculate sim/future states

function X2_sim = Delay_no_kron_Forward_Rollout(dynamics_matrices, xic, nx, U_traj, p)
% Based on some initial condition for state (this may have a time delay
% embedded OR not) and a control input trajectory, this outputs an X_sim+ 
% state w/o time embedding
% nx-> meant to be the original state vector dimension (pre any embedding)

% length of the trajectory from cols
nm = length(U_traj(1,:));
nu = length(U_traj(:,1));

% init a place to store solns
X1_sim = zeros((p+1)*nx, nm); % ea col of x_k
X2_sim = zeros(nx, nm); % ea col of x_k+1 to be returned

% build a C matrix that maps y_k to x_k
C = [eye(nx), zeros(nx, p*nx)];

% set up the initial condition
X1_sim(:, 1) = xic;

% unpack the dynamics matrices into easier structures to iterate through
B_tilde = dynamics_matrices(1:nx, 1:nu);
A_tilde = dynamics_matrices(1:nx, nu+1:end);

% flow dynamics through and propogate answers to time embedded sim matrix
% up through the second to last column of X (ie nm-1 in this function)
for k=1:nm    
    % x_k+1 = B*u_k + A*x_k
    X2_sim(:,k) = B_tilde*U_traj(:,k) + A_tilde*(X1_sim(:,k)); 

    % x_k+1 += B*u_k
%     X2_sim(:,k) = X2_sim(:,k) + B_tilde*U_traj(:,k);
%     for i = 1:p+1
%         % A_1 => cols 2:4, A_2 => cols 5:7, A_3 => cols 8:10, etc
%         start_idx = nu+1+(i-1)*nx;
%         A_i = dynamics_matrices(1:nx, start_idx:start_idx+nx-1);
%         
%         % x_k rows 1:3, x_k-1 rows 4:6, x_k-2 rows 7:9, etc
%         start_idx = (i-1)*nx+1; %same as before but shifted 1 step forward
% 
%         % x_k+1 += A1*x_k + A2*x_k-1 + A3*x_k-2 + ...
%         X2_sim(:,k) = X2_sim(:,k) + A_i*X1_sim(start_idx:start_idx+nx-1, k);
%     end
    
    % x_k+1 is now solved for, need to build up the subsequent column of
    % X_k
    if(k < nm)
        X1_sim(1:nx, k+1) = X2_sim(:,k);
        X1_sim(nx+1:end, k+1) = X1_sim(1:p*nx, k);
    end
    
end

% need to solve the terminal term
% x_m += B*u_m-1
%X2_sim(:,end) = B_tilde*U_traj(:,end) + A_tilde*(X1_sim(:,end));

% dumb way to get final term
% for i = 1:p
%     % A_1 => cols 2:4, A_2 => cols 5:7, A_3 => cols 8:10, etc
%     start_idx = nu+1+(i-1)*nx;
%     A_i = dynamics_matrices(1:nx, start_idx:start_idx+nx-1);
%     
%     % x_k rows 1:3, x_k-1 rows 4:6, x_k-2 rows 7:9, etc
%     start_idx = (i-1)*nx+1; %same as before but shifted 1 step forward
% 
%     % x_k+1 += A1*x_m-1 + A2*x_m-2 + A3*x_m-3 + ...
%     X2_sim(:,end) = X2_sim(:,end) + A_i*X1_sim(start_idx:start_idx+nx-1, end);
% end

end

%% originally came from this single delay example that I generalized:
% A_1 = gen_dyn_matrices_delays{1,1}(:,2:4);
% A_2 = gen_dyn_matrices_delays{1,1}(:,5:7);
% B_1 = gen_dyn_matrices_delays{1,1}(:,1);
% 
% % blank matrices to store the simulated rollout
% X_k2_sim = zeros(length(gen_X_k2_delays{1,1}(:,1)), length(gen_X_k2_delays{1,1}(1,:)));
% X_k1_sim = zeros(length(gen_X_k1_delays{1,1}(:,1)), length(gen_X_k1_delays{1,1}(1,:)));
% 
% % initial conditions
% x_ic = gen_X_k1_delays{1,1}(:,1);
% X_k1_sim(:,1) = x_ic;
% 
% % still ironing out this rollout
% for i=1:length(gen_X_k1_delays{1,1}(1,:))-1
%     
%     X_k2_sim(:,i) = A_1*X_k1_sim(1:3, i) + A_2*X_k1_sim(4:6, i)...
%         + B_1*gen_U_k1_delays{1, 1}(:,i);
%     X_k1_sim(1:3,i+1) = X_k2_sim(:,i); % iterate the next up x_k
%     X_k1_sim(4:6,i+1) = X_k1_sim(1:3,i); % shift the prev entry down a set of rows
%     %pretty sure this leaves the final column of x_k2 blank
% end