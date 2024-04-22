%% Building out Time Delay Embedded Matrices

function Y = Time_Delay_Embed(X, p)
% takes in a matrix X and a desired number of time delays to then build out
% a Hankel style matrix that expands the system to embed the delays.
%
% Returns a column-wise time delay embedding for any matrix and number of
% delays. I've not currently programmed in a way to appropriately handle
% someone erroneously passing in 0 time delays

nm = length(X(1,:)); % number of measurements (ie cols in X)
n_states = length(X(:,1)); % dimensionality of each entry (ie num rows)

% each col would be:
% yk = [xk;
%       xk-1;
%       xk-2;
%       .
%       .
%       xk-p]

% Overall this makes the Hankel matrix (using p = 1 as an ex):
% Y = [x1, x2, x3, .... xm;
%      x0, x1, x2, .... xm-1]
% (for entries that are pre-x1, plan is to couch it the matrix with x1s)

% More generally:
% Y = [x1,   x2,   x3,   x4, ...   xm;
%      x0,   x1,   x2,   x3, ...   xm-1;
%      ..,   x0,   x1,   x2, ...   xm-2;
%      ................................;
%      x1-p, x2-p, x3-p, x4-p, ... xm-p]

% declare / size the time embedded matrix
Y = zeros(n_states*(1+p), nm);

% first batch of rows (ie 1:n_states) of Y don't have a delay
Y(1:n_states, :) = X(:, :);

% iterate starting from next row down.
%for i=n_states+1:n_states:(1+p)*n_states
for i=1:p
    for j=1:nm
        start_row = i*n_states+1; % actual starting index row in Y we're filling
        end_row = start_row+n_states-1; % actual ending index row in Y being filled
        
        % general case where the time delay is fine
        if(j - i)>0
            Y(start_row:end_row, j) = X(:, j-i);
        % edge case where we're close to the start of the matrix
        % time delay is going too far back/out of index. couch with x1.
        % akin to MPC when we exceed the time horizon and just perpetuate
        % the last known state
        else
            Y(start_row:end_row, j) = X(:, 1);
        end
    end
end

end