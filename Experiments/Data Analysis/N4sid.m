dt = readtable("Koopman Data/Modified_Data.xlsx");
u = dt{:,{'commanded_closure_pressure_psi','commanded_x_mm','commanded_y_mm'}};
y =  dt{:,{'P_jaw1_psi','P_jaw2_psi','P_jaw3_psi'}};
data = iddata(y,u,1/16)
nx = 1:20;
sys = n4sid(data,nx,'Ts',1/16);

%compare(data,sys)
co = ctrb(sys)
disp("Rank of controllability matrix: ")
disp(rank(co))
disp("uncontrollable states:")
disp(length(sys.A)-rank(co))
% Check results

%Look at svd of hankel matrix
hsv = hsvd(sys)