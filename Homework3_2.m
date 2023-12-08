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
t = 0:1 / fs:6 - (1 / fs);
f1 = 4;
f2 = 8;
f3 = 12;
t1 = 0:1 / fs:2 - (1 / fs);
t2 = 2:1 / fs:4 - (1 / fs);
t3 = 4:1 / fs:6 - (1 / fs);
x = [cos(2 * pi * f1 * t1) cos(2 * pi * f2 * t2) cos(2 * pi * f3 * t3)];
b = [0.969531 -1.923772 0.969531];
a = [1 -1.923772 0.939063];
filtered_signal = filter(b, a, x);
%%%
% Now we plot filtered signals
figure('Name', 'filtered signal delta_f=4 Hz');
plot(t, filtered_signal);
grid on;
title('Filtered Signal with delta_f =4 Hz');
xlabel('n');
ylabel('amplitude');
x_0 = [1 zeros(1, 49)];
filtered_signal = filter(b, a, x_0);
figure('Name', 'impulse response delta_f=4 Hz');
stem(filtered_signal);
grid on;
title('impulse response with delta_f =4 Hz');
xlabel('n');
ylabel('amplitude');
b1 = [0.996088 -1.976468 0.996088];
a1 = [1 -1.976468 0.992177];
filtered_signal1 = filter(b1, a1, x);
figure('Name', 'filtered signal delta-f=0.5 Hz');
plot(t, filtered_signal1);
grid on;
title('Filtered Signal with delta-f =0.5 Hz');
xlabel('n');
ylabel('amplitude');

x_0 = [1 zeros(1, 49)];
filtered_signal = filter(b1, a1, x_0);
figure('Name', 'impulse response delta_f=0.5 Hz');
stem(filtered_signal);
grid on;
title('impulse response with delta_f =0.5 Hz');
xlabel('n');
ylabel('amplitude');

%% Homework2

% Here we add some essential variables
fs = 400;
t = 0:1 / fs:6 - (1 / fs);
f1 = 4;
f2 = 8;
f3 = 12;
N = 2400;
n = 0:N - 1;
f0 = 8;
t1 = 0:1 / fs:2 - (1 / fs);
t2 = 2:1 / fs:4 - (1 / fs);
t3 = 4:1 / fs:6 - (1 / fs);
w0 = 2 * pi * f0 / fs;

delta_f = 4;
delta_w = 2 * pi * delta_f / fs;
beta = tan(delta_w / 2);
x_delta = zeros(1, N + 1);
x_delta(1) = 1;

x = [cos(2 * pi * f1 * t1) cos(2 * pi * f2 * t2) cos(2 * pi * f3 * t3)];
b1 = [1 -2 * cos(w0) 1] / (beta + 1);
a1 = [-2 * cos(w0) / (1 + beta) (1 - beta) / (1 + beta)];
w2 = 0;
w1 = 0;
x2 = 0;
x1 = 0;
y = zeros(1, N);
%%%
% Here we do the diffrential equation
for i = 1:N
    y(i) = -a1(1) * w1 -a1(2) * w2 + b1(1) * x(i) + b1(2) * x1 + b1(3) * x2;
    w2 = w1;
    w1 = y(i);
    x2 = x1;
    x1 = x(i);
end

%%%
% Now we plot the impulse responses
figure('Name', 'Impulse Responses of the Two Notch Filters');
plot(n, y, "LineWidth", 1.5);
title("Impuulse response of Filter H1 in Time Domain (\Deltaf = 4Hz)");
xlabel("n");
ylabel("Amplitude");
xlim([0 2400]);
grid on;
