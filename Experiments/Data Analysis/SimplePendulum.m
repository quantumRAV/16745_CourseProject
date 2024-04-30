clear all
close all
%% Simple Pendulum. 
% Discretize continuous-time dynamics using RK3 and roll out the states
% Note: theta = 0 corresponds to the pendulum hanging straight down. theta
% = pi/2 correspends to the pendulum pointing to the right

m = 0.1; %kg
g = 9.81; %m/s^2
l = 0.3; %m

dt = 0.05; %time step in seconds
N = 1000; %total number of steps to simulate
t = [1:N]*dt;

x0 = [pi/2;0]; %zero initial conditions. 1st element is the position, 2nd is the velocity
u = zeros(1,N); %No applied torque

%simulate using RK4
x_RK4 = zeros(2,N);
x_RK4(:,1) = x0; %initial condition

pen_dyn = @(x,u) pendulum_dynamics_continuous(x,u,m,g,l);

for ii=1:N-1
    x_RK4(:,ii+1) = RK4(pen_dyn, x_RK4(:,ii), u(ii), dt);  
end

figure()
subplot(2,1,1)
hold on
plot(t,x_RK4(1,:),'r--', DisplayName = "Position");
xlabel('time (s)')
ylabel('Angular position $\theta$ (rad)', 'Interpreter','latex')

subplot(2,1,2)
hold on
plot(t,x_RK4(2,:),'r--', DisplayName = "Angular Velocity");
xlabel('time (s)')
ylabel('Angular velocity, $\dot{\theta}$ (rad/s))', 'Interpreter','latex')

%% fit linear model to pendulum data using N4sid, and compare the A and B matrices
figure()
nx = 1:5; %model order
sys = n4sid(u.',x_RK4(1,:).', nx, 'Ts', dt);

compare(u.',x_RK4(1,:).',sys);
hold on
plot(t,x_RK4(1,:),'r--', DisplayName = "Angular position from RK4");

%Display the A, B and C matrices. Without forcing (i.e. u = 0), we would
%expect the B matrix to be close to 0

disp("A:")
disp(sys.A)

disp("B:")
disp(sys.B)

disp("C:")
disp(sys.C)


%% Fit using homebrew dynamics regression


function x_dot = pendulum_dynamics_continuous(x,u,m,g,l)
    x_dot=zeros(2,1);
    x_dot(1) = x(2); %velocity kinematics
    x_dot(2) = -(g/l)*sin(x(1)) + m*(l^2)*u; %continuous pendulum dynamics
end

function xk_1 = RK4(f, x, u, dt)

    k1 = f(x,u);
    k2 = f(x + 0.5*dt*k1,u);
    k3 = f(x + 0.5*dt*k2,u);
    k4 = f(x + dt*k3,u);

    xk_1 = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4); %next state

end

