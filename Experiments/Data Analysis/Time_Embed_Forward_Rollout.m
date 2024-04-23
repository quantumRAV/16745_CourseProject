%% Use A_tilde and B_tilde matrices to calculate sim/future states

function X_sim = Time_Embed_Forward_Rollout(A_tilde, B_tilde, xic, nx, U_traj, p)
% Based on some initial condition for state (this may have a time delay
% embedded OR not) and a control input trajectory (similarly may or may 
% not have time embedding), this outputs an X_sim+ state w/o time embedding
% nx-> meant to be the original state vector dimension (pre any embedding)

% length of the trajectory from cols
nm = length(U_traj(1,:));

% init a place to store solns
Y_sim = zeros((p+1)*nx, nm+1); % ea col of y_k (and subsequently y_k+1)
X_sim = zeros(nx, nm); % ea col of x_k+1 to be returned

% build a C matrix that maps y_k to x_k
C = [eye(nx), zeros(nx, p*nx)];

% set up the initial condition
Y_sim(:, 1) = xic;

for k=1:nm
    % iterate a dynamics step for y_k+1
    Y_sim(:,k+1) = A_tilde*Y_sim(:, k) + B_tilde*U_traj(:,k);
    % mapping of y_k+1 to x_k+1 (recall that X_sim is returned as X+)
    X_sim(:,k) = C*Y_sim(:,k+1);
end

end