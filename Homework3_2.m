%% *Homework3_2*
%% Programmers
% Mohammad MAhdi Elyasi - 9823007
%
% Moein Nasiri - 9823093
%% Clear Workspace
close all;
clear;
clc;
%% Homework1
% Here we declare some essential variables
fs = 400;
t = 0:1 / fs:6-(1/fs);
f1 = 4;
f2 = 8;
f3 = 12;
t1 = 0:1 / fs:2-(1/fs);
t2 = 2:1 / fs:4-(1/fs);
t3 = 4:1 / fs:6-(1/fs);
x=[cos(2 * pi * f1 * t1) cos(2 * pi * f2 * t2) cos(2 * pi * f3 * t3)];
b = [0.969531 -1.923772 0.969531];
a = [1 -1.923772 0.939063];
filtered_signal = filter(b, a, x);
%%%
% Now we plot filtered signals
figure('Name','filtered signal delta_f=4 Hz');
plot(t, filtered_signal);
grid on;
title('Filtered Signal with delta_f =4 Hz');
xlabel('n');
ylabel('amplitude');
x_0 = [1 zeros(1,49)];
filtered_signal = filter(b, a, x_0);
figure('Name','impulse response delta_f=4 Hz');
stem( filtered_signal);
grid on;
title('impulse response with delta_f =4 Hz');
xlabel('n');
ylabel('amplitude');
b1=[0.996088 -1.976468 0.996088];
a1=[1 -1.976468 0.992177];
filtered_signal1 = filter(b1, a1, x);
figure('Name','filtered signal delta-f=0.5 Hz');
plot(t, filtered_signal1);
grid on;
title('Filtered Signal with delta-f =0.5 Hz');
xlabel('n');
ylabel('amplitude');

x_0 = [1 zeros(1,49)];
filtered_signal = filter(b1, a1, x_0);
figure('Name','impulse response delta_f=0.5 Hz');
stem( filtered_signal);
grid on;
title('impulse response with delta_f =0.5 Hz');
xlabel('n');
ylabel('amplitude');

