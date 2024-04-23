%% Linear Regression Fitting A and B Matrices to Data

function [A, B] = Dynamics_Mat_Reg(X_k, U_k, X_k2, nx, nu)

% intialize A and B matrices to be returned    
A = zeros(nx, nx);
B = zeros(nx, nu);

% need an identity matrix for the kronecker products
I = eye(nx, nx);

% used to calculate qty of columns for the large set of kronecker products
nm = length(X_k);

kron_col_x = nx*nx;
kron_col_u = nu*nx;

% initialize the large kronecker matrix
big_kron = zeros(nx*nm, (kron_col_x + kron_col_u));

% initialize a place to gather up the x_k+1 values (ie x_k2)
x_k2_vector = zeros(nx*nm, 1); 

% fill the large kronecker matrix
for k=1:nm % need to do some careful indexing here
    x_start_pt = (k-1)*(kron_col_x+kron_col_u)+1;
    u_start_pt = x_start_pt+kron_col_x;

    big_kron(nx*(k-1)+1:nx*k, 1:kron_col_x) = kron(X_k(:,k)',I); 
    big_kron(nx*(k-1)+1:nx*k, kron_col_x+1:kron_col_x+kron_col_u) = kron(U_k(:,k)',I);

    % in this iter, probably can build up the sum of x_k2s
    x_k2_vector(nx*(k-1)+1:nx*k) = X_k2(:,k);
end

% pseudo inverse the summation of values over time with the kronecker
% matrix to solve for least squares answer for a vectorized version of A
% and B
stacked_vectorized_matrices = big_kron\x_k2_vector;

% trying some different inverses
% apparently this comes out more equitably distributed than \ which
% prioritizes zero elements
moore_penrose_stacked_matrix = pinv(big_kron)*x_k2_vector;

% sb same answer as moores-penrose, but per matlab doc it's "typically"
% more efficient
ls_stacked_matrix = lsqminnorm(big_kron, x_k2_vector);

% Use the first entries to extract/fill A
for i=1:nx
    for j=1:nx
        %A(i, j) = stacked_vectorized_matrices(i*j);
        %A(i, j) = moore_penrose_stacked_matrix(i*j);
        A(i, j) = ls_stacked_matrix(i*j);
    end
end

% Use the next set of entries to extract/fill B
for i = 1:nx
    for j=1:nu
        % need to offset/shift to skip over the vec(A) entries
        %B(i, j) = stacked_vectorized_matrices(i*j + nx*nx);
        %B(i, j) = moore_penrose_stacked_matrix(i*j + nx*nx);
        B(i, j) = ls_stacked_matrix(i*j + nx*nx);
    end
end

end