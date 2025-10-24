% Dynamic Euler–Bernoulli beam: finite-difference + ode45
clear; clc; close all;

% Parameters
L   = 10; EI = 20.83; rhoA = 1; q0 = 0.015;

% Discretization
Nx = 50;
x  = linspace(0, L, Nx)';     
dx = x(2)-x(1);

% Fourth-derivative matrix
e = ones(Nx,1);
D4 = spdiags([e -4*e 6*e -4*e e], -2:2, Nx, Nx) / dx^4;

% Boundary conditions: u=0, u''=0 at both ends
keep = 3:Nx-2;
D4 = D4(keep, keep);
x_in = x(keep);
Nx_in = numel(x_in);

% Matrices
K = EI * full(D4);
M = rhoA * eye(Nx_in);

% Load
qfun = @(t) q0 * sin(pi*x_in/L) * sin(pi*t);

% System
f = @(t,y)[ y(Nx_in+1:end);
            M\(qfun(t) - K*y(1:Nx_in)) ];

% Initial conditions and integration
y0 = zeros(2*Nx_in,1);
options = odeset('OutputFcn',@(t,y,flag) myOutputFcn(t,y,flag,x_in));
[t, y] = ode45(f, [0 4], y0, options);

% Fields
U = y(:,1:Nx_in);                      % displacement
Q = zeros(length(t), Nx_in);           % load
for k = 1:length(t)
    Q(k,:) = qfun(t(k));
end

% --- Combined 3D plot ---
figure('Position',[100 100 800 500]); hold on;
surf(x_in, t, U, 'FaceColor',[0.1 0.4 0.9], 'FaceAlpha',0.8, 'EdgeColor','none');  % blueish
surf(x_in, t, Q, 'FaceColor',[0.9 0.3 0.3], 'FaceAlpha',0.6, 'EdgeColor','none');  % reddish

xlabel('x [m]'); ylabel('t [s]');
zlabel('Amplitude [m or N/m]');
title('Beam deflection u(x,t) and applied load q(x,t)');
legend({'u(x,t) - deflection','q(x,t) - load'}, 'Location','northoutside','Orientation','horizontal');
view(45,35); grid on; box on;




function status = myOutputFcn(t,y,flag,x_in)
    persistent deflection load
    L   = 10;
    q0 = 0.015;
    q = q0 .* sin(pi*x_in/L) .* sin(pi*mean(t));
    q = [0, q', 0];
    x = [0, x_in', L];
    switch flag  
        case 'init'
            figure
            xlabel('x [m]')
            ylabel('Amplitude [m or N/m]')
            ylim([-q0, q0]);
            deflection = animatedline('Color','b');
            load = animatedline('Color','r');
            legend({'u(x,t) - deflection','q(x,t) - load'})
        case []
            for i=1:length(x_in)
                u(i)=y(i);
            end
            u = [0, u, 0];
            clearpoints(deflection)
            addpoints(deflection,x,u)
            clearpoints(load)
            addpoints(load,x,q)
            drawnow limitrate
        case 'done'
    end
    status = 0;
end