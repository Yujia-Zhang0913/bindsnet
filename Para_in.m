% Pressure source
p_source = 8e5; % Pa

% 4-way directional valve
valve_area_max = 3e-5; % m^2
L0 = 1e-5;

% Pipes
L_pipe = 20; % m
flow_area = 1e-4; % m^2

% Air muscle actuators
L_muscle = 30; % cm
L_braid = 32; % cm
n = 2.5;

% Flow control valves
dp_crack = 1e4; % Pa
dp_max = 2e4; % Pa
area_max = 1e-5; % m^2
area_leak = 1e-10; % m^2
area_restrict =1e-6; % m^2 

% Translational dampers
D_translation = 1; % N/(m/s)

% Translational hard stops
x_bound = 8; % cm

% Linkage
L_top = 15; % cm
L_bottom = 15; % cm

% Load
inertia = 0.005; % kg*m^2
D_rotation = 10; % N*m/(rad/s) 
K_spring = 6.4; % N*m/rad
theta_upper = 0.4; % rad
theta_lower = 0; % rad

% Environment
T_atm = 293.15; % K
p_atm = 0.101325; % MPa
network=double(0.0);
anti_network = double(0.0);
tout = [0;0];