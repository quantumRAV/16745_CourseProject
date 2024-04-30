%% Building out Time Delay Embedded Matrices

function Y = Time_Delay_Embed(X, p)
% takes in a matrix X and a desired number of time delays to then build out
% a Hankel style matrix that expands the system to embed the delays.
%
% Returns a column-wise time delay embedding for any matrix and number of
% delays. I've not programmed in a way to protect if someone tries to use
% too many delays (ie p>nm).
% I've not currently programmed in a way to appropriately handle
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
% Y = [x2, x3, x4, .... xm;
%      x1, x2, x3, .... xm-1]

% More generally:
% Y = [x1+p,   x2+p,   x3+p,   x4+p,  ... xm-p+p;
%      x1+p-1, x2+p-1, x3+p-1, x4+p-1,... xm-p+p-1;
%      .................................. xm-p+p-2;   
%      x3,     x4,     x5, .............. ;
%      x2,     x3,     x4, .............. ;
%      x1,     x2,     x3,     x4-p, ..., xm-p]

% declare / size the time embedded matrix
Y = zeros(n_states*(1+p), nm-p);

% first batch of rows (ie 1:n_states) of Y don't have a delay
%Y(1:n_states, :) = X(:, :);

% iterate starting from next row down.
%for i=n_states+1:n_states:(1+p)*n_states
for i=1:p+1
    for j=1:nm-p
        start_row = (i-1)*n_states+1; % actual starting index row in Y we're filling
        end_row = start_row+n_states-1; % actual ending index row in Y being filled
        Y(start_row:end_row, j) = X(:, j+p-i+1);
    end
end

end