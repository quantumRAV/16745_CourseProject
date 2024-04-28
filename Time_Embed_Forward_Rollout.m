%% Use A_tilde and B_tilde matrices to calculate sim/future states

function X2_sim = Time_Embed_Forward_Rollout(dynamics_matrices, xic, nx, U_traj, p)
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

for k=1:nm
    % x_k+1 += B*u_k
    X2_sim(:,k) = X2_sim(:,k) + B_tilde*U_traj(:,k);
    for i = 1:p
        % A_1 => cols 2:4, A_2 => cols 5:7, A_3 => cols 8:10, etc
        start_idx = nu+1+(i-1)*nx;
        A_i = dynamics_matrices(1:nx, start_idx:start_idx+nx-1);
        
        % x_k rows 1:3, x_k-1 rows 4:6, x_k-2 rows 7:9, etc
        start_idx = (i-1)*nx+1; %same as before but shifted 1 step forward

        % x_k+1 += A1*x_k + A2*x_k-1 + A3*x_k-2 + ...
        X2_sim(:,k) = X2_sim(:,k) + A_i*X1_sim(start_idx:start_idx+nx-1, k);
    end
    Y_sim(:,k+1) = A_tilde*Y_sim(:, k) + B_tilde*U_traj(:,k);
    % mapping of y_k+1 to x_k+1 (recall that X_sim is returned as X+)
    X_sim(:,k) = C*Y_sim(:,k+1);
end

end