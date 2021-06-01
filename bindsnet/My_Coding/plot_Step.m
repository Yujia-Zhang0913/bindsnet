if ~exist('final_analysis', 'var') || ...
        ~isgraphics(final_analysis, 'figure')
    final_analysis = figure('Name', 'final_analysis');
end

figure(final_analysis);
clf(final_analysis);
input = 2.4 + sin(0.2*valuess.time'-pi/2)*2.4;

plotLengthPressure(valuess,pos',input)


function plotLengthPressure(simlog,pos,input)
% Get simulation results
data = simlog.get(1);
pos_6 = data.get(6).Values.Data;
t = simlog.time';

anti_network = simlog.signals(2).values';
network = simlog.signals(6).values';
des_anti_pressure = simlog.signals(3).values';
des_pressure = simlog.signals(4).values';
% Plot results

simlog_handles(1) = subplot(4, 1, 1);
plot(t, pid_out, 'LineWidth', 1)
grid on
subtitle('Input and Output')
legend("pidout")
ylabel('pidout')
xlabel('Time (s)')

simlog_handles(2) = subplot(4, 1, 2);
subtitle('networkout')
plot(t, anti_network, 'LineWidth', 1)
hold on
plot(t, network, 'LineWidth', 1)
hold off
grid on
legend('antinetwork', 'network')
ylabel('networkout')
xlabel('Time (s)')

simlog_handles(3) = subplot(4, 1, 3);
subtitle('pressure')
plot(t, des_anti_pressure, 'LineWidth', 1)
hold on
plot(t, des_pressure, 'LineWidth', 1)
hold off
grid on
legend('desantipressure', 'despressure')
ylabel('Pressure(Pa)')
xlabel('Time (s)')

simlog_handles(4) = subplot(4, 1, 4);
subtitle('pos input and error')
plot(t, pos, 'LineWidth', 1)
hold on
plot(t, input, 'LineWidth', 1)
plot(t, input-pos, 'LineWidth', 1)
hold off
grid on
legend('pos', 'input','error')
ylabel('workout(Degree)')
xlabel('Time (s)')
linkaxes(simlog_handles, 'x')

end
