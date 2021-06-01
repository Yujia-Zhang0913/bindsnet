data = ff.get(1);
data2 = ff.get(2);
error = data.get(1).Values.data;
error2 = data2.get(1).Values.data;
My_sum(1) = 0;
My_sum2(1) = 0;
for t=1:length(error)
    My_sum(t+1) = My_sum(t) + error(t)^2;
end
My_sum = My_sum/length(error);
time = 0:0.052:50;
subplot(2,1,1)
plot(time(1,1:length(error)+1),My_sum(1,1:length(error)+1),'LineWidth',3);
for t=1:length(error2)
    My_sum2(t+1) = My_sum2(t) + error2(t)^2;
end
My_sum2 = My_sum2/length(error2);
t = 0:0.0014:50;
hold on
plot(t(1,1:length(error2)+1),My_sum2(1,1:length(error2)+1),'LineWidth',3);
grid on
legend("PID","0.1 PID+feedforward");
ylabel('MSE')
xlabel('Time (s)')
title("Mean Square Error Per Dot for 0.1 PID+feed-forward ")
subplot(2,1,2)
data = Sin.get(4);
data2 = Sin.get(1);
error = data.get(1).Values.data;
error2 = data2.get(1).Values.data;
My_sum(1) = 0;
My_sum2(1) = 0;
for t=1:1608
    My_sum(t+1) = My_sum(t) + error(t)^2;

end
My_sum = My_sum/1609;
time = 0:0.03:50;
plot(time(1,1:1609),My_sum,'LineWidth',3);
for t=1:34320
    My_sum2(t+1) = My_sum2(t) + error2(t)^2;
end
My_sum2 = My_sum2/34320;
t = 0:0.0014:50;
hold on
plot(t(1,1:34321),My_sum2,'LineWidth',3);
grid on
legend("PID","PID+feedforward");
ylabel('MSE')
xlabel('Time (s)')
title("Mean Square Error Per Dot for PID+feed-forward")