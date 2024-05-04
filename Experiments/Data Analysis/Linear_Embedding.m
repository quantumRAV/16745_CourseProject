format long

%% For the 1st set of data collected in early April, rigid 40 mm cylinder
dt = readtable("Koopman Data/Koopman_Testing_06_04_2024_10_04_48_rigid40.xlsx");


u = dt{:,{'commanded_closure_pressure_psi','commanded_x_mm','commanded_y_mm'}};
y =  dt{:,{'P_jaw1_psi','P_jaw2_psi','P_jaw3_psi'}};


nx = 1:20;
[sys, x0] = n4sid(u,y,nx,'Ts',1/16);

figure()
compare(u,y,sys)

co = ctrb(sys)
disp("Rank of controllability matrix: ")
disp(rank(co))
disp("uncontrollable states:")
disp(length(sys.A)-rank(co))

obv = obsv(sys);
disp("Rank of observability matrix: ")
disp(rank(obv))

% Check svd
hsv = hsvd(sys)



%% For the 2nd set of data collected on April 30th, just in contact, rigid 40 mm cylinder

dt = readtable("Koopman Data/Koopman_Testing_30_04_2024_17_07_00_modified_40mm_rigid_contact.xlsx");


u = dt{:,{'commanded_closure_pressure_psi','commanded_x_mm','commanded_y_mm'}};
y =  dt{:,{'P_jaw1_psi','P_jaw2_psi','P_jaw3_psi'}};


nx = 1:20;
[sys, x0] = n4sid(u,y,nx,'Ts',1/16);

figure()
compare(u,y,sys);

[ymod,fit,ic] = compare(u,y,sys);

co = ctrb(sys)
disp("Rank of controllability matrix: ")
disp(rank(co))
disp("uncontrollable states:")
disp(length(sys.A)-rank(co))


obv = obsv(sys);
disp("Rank of observability matrix: ")
disp(rank(obv))

% Check svd
hsv = hsvd(sys)

% Initial condition
sys.x0

%% Forward Simulate with A and B and C matrices
N = size(y,1);
ysim = zeros(size(y)).'; %no x N matrix
y_actual = y.'; %the actual data. Will use this to correct for observation error
ysim(:,1) = y_actual(:,1);

xsim = zeros(length(sys.x0),N); %nx x N matrix
%xsim(:,1) = pinv(sys.C)*ysim(:,1);
xsim(:,1) = sys.x0; %with this initial condition, the simulated and matlab responses exactly overlap


usim = u.'; 
t = (1/16)*[1:1:N]; 

useObserverFeedback = false;

for ii =1:(N-1)
    xsim(:,ii+1) = sys.A*xsim(:,ii) + sys.B*usim(:,ii); 
    
    if (useObserverFeedback == true & ii>1)
        xsim(:,ii+1) = xsim(:,ii+1) + sys.K*(y_actual(:,ii)-ysim(:,ii)); %add state observer feedback
    end
    ysim(:,ii+1) = sys.C*xsim(:,ii+1); %simulated y position
end

% Matlab simulated response
y_matlab = lsim(sys,u,t,sys.x0); %using the system estimated initial condition
y_matlab_ic = lsim(sys,u,t,ic); %initial condition from compare(...). Produces close match to actual data with observer feedback

figure()
hold on
plot(t,ysim(1,:),'b-');
plot(t,y_matlab(:,1),'r--')
plot(t,y_actual(1,:),'m-.')
legend();

%% For the 3rd set of data collected on May3rd, just in contact, elastic Ecoflex 00-30 40 mm cylinder

dt = readtable("Koopman Data/Koopman_Testing_03_05_2024_19_30_51_Elastic40.xlsx");


u = dt{:,{'commanded_closure_pressure_psi','commanded_x_mm','commanded_y_mm'}};
y =  dt{:,{'P_jaw1_psi','P_jaw2_psi','P_jaw3_psi'}};


nx = 1:20;
[sys, x0] = n4sid(u,y,nx,'Ts',1/16);

figure()
compare(u,y,sys);

[ymod,fit,ic] = compare(u,y,sys);

co = ctrb(sys)
disp("Rank of controllability matrix: ")
disp(rank(co))
disp("uncontrollable states:")
disp(length(sys.A)-rank(co))


obv = obsv(sys);
disp("Rank of observability matrix: ")
disp(rank(obv))

% Check svd
hsv = hsvd(sys)

% Initial condition
sys.x0
