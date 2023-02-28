clc;
e_meas_rms = zeros(100,1);
e_x_rms = zeros(100,1);
e_v_rms = zeros(100,1);
c = 1:100;
for m=1:100
    dt = 0.1; % time interval
    t = 0:0.1:1000; % time
    A = [0 1; 0 0]; % system dynamics
    F = eye(size(A))+ dt*A; % state transition 
    B = zeros(); % no control G=0 
    C = [1 0]; % measurement H=C
    D = 0; % no control feedback
    wk = 0.1; % flight state variance
    P = [500 0; 0 -0.350]; % initial error covariance
    q = [(dt^4)/4 (dt^3)/2; (dt^3)/2 (dt)^2]; % randomness in acceleration
    Q = wk*q; % process noise covariance
    vk = (0.01) * randn(length(t), 1); % measurement error 
    %R =  mean(vk * transpose(vk)); % measurement noise covariance based on
    %fornula
    R = (0.01)^2; % generic std deviation squared
    x = zeros(2, length(t)); % state
    x_hat = zeros(2, length(t)); % state estimate
    y = zeros(); % measurement
    y_hat = zeros(); % measurement estimate
    x_hat(:,1) =[0;0]; % initial state estimate
    x(:,1) = [540; -0.300]; % initial true state
    y(1) = C*x(:,1)+vk(1); % initial measurement
    y_hat(1) = C*x_hat(:,1); % initial measurement estimate
    e_y = zeros(); % error
    e_y(1) = y(1) - y_hat(1);
    e_x1 = zeros();
    e_x1(1) = x(1,1) - x_hat(1,1);
    e_x2 = zeros();
    e_x1(2) = x(2,1) - x_hat(2,1);

    for i=2:length(t)
        x(:,i) = F*x(:,i-1);
        y(i) = C*x(:,i)+vk(i);
        % time update
        x_hat(:,i) = F*x_hat(:,i-1);  % x[n+1|n]
        P = F*P*F' + Q; % P[n+1|n]    

        y_hat(i) = C*x_hat(:,i);
        e_y(i) = y(i) - y_hat(i); 

        % measurement update
        k = P*C'/(C*P*C' + R); % gain
        x_hat(:,i) = x_hat(:,i) + k*e_y(i); % x[n|n]
        P = (eye(2) - k*C)*P; % P[n|n]
        e_x1(i) = x(1, i)-x_hat(1, i);
        e_x2(i) = x(2, i)-x_hat(2, i);

    end
    e_meas_rms(m) = rms(e_y);
    e_x_rms(m) = rms(e_x1);
    e_v_rms(m) = rms(e_x2)*1000;
    
end
subplot(2,1,1);
plot(t, x_hat(1,:), ".r", t, x(1,:), "g");
title('Kalman Filter Design');
xlabel('Time (s)');
ylabel('Distance (km)');
ylim([0 1000]);
legend('Filtered', 'True');
grid on;
subplot(2,1,2);
plot(t, x_hat(2,:)*(-1000), ".r", t, x(2,:)*(-1000), "g");
%ylim([-10 10]);
xlabel('Time (s)');
ylabel('Speed (m/s)')
legend('Filtered', 'True');
grid on;
figure();
subplot(3,1,1);
plot(c, e_meas_rms, "LineWidth", 1.5);
title('Monte Carlo RMS Estimated Measurement Error');
xlabel('Trial No.');
ylabel('RMS Error (km)','VerticalAlignment','bottom');
grid on;
subplot(3,1,2);
plot(c, e_x_rms, "LineWidth", 1.5);
title('Monte Carlo RMS Distance Error');
xlabel('Trial No.');
ylabel('RMS Error (km)','VerticalAlignment','bottom');
grid on;
subplot(3,1,3);

plot(c, e_v_rms, "LineWidth", 1.5);
title('Monte Carlo RMS Speed Error');
xlabel('Trial No.');
ylabel('RMS Error (m/s)','VerticalAlignment','bottom');
grid on;