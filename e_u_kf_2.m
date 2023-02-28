n = 6;
T = 100;
x_rmsError_ukf = zeros(n,T);
x_rmsError_ekf = zeros(n,T);
for m=1:100
    dt = 0.1;     % seconds
    Q = diag([0.0, 0.0, 0.0, 0.01^2, 0.01^2, 0.01^2]);
    R = diag([0.05, 0.05, 0.05]);
    t = 0;
    f=@(x,t)[x(1)+(x(5)*sin(x(3))+x(6)*sin(x(3)))*dt;x(2)+((x(5)*cos(x(3))-x(6)*sin(x(3)))/cos(x(1)))*dt;x(3)+(x(4)-tan(x(1))*(x(5)*cos(x(3))-x(6)*sin(x(3))))*dt; 0.05*sin(0.1*t); 0.05*sin(0.2*t); 0.05*sin(0.3*t)];
    h=@(x,t)[x(4); x(5); x(6)];
    s=[180/4; 0.0; 0.0; 0.0; 0.0; 0.0];
    x=s+sqrt(Q)*randn(6,1);
    S=zeros(n);
    simTime = 600; % seconds
    tspan = 0:dt:simTime;
    N = length(tspan);
    x_ukf = zeros(n,N);
    x_ekf = zeros(n,N);
    error_ukf = zeros(n,N);
    error_ekf = zeros(n,N);
    s_True = zeros(n,N);
    zV = zeros(3,N);

    %% Unscented Kalman Filter
    for k=1:N
        if k~=1
            t=t+dt;
        end
        z = h(s,t) + sqrt(R)*randn(3,1);
        s_True(:,k) = s;
        zV(:,k) = z;
        [x, S] = ukf(f,x,S,h,z,Q,R,t);
        x_ukf(:,k) = x;
        s = f(s,t) + sqrt(Q)*randn(6,1);
    end
    for k=1:6  
        %figure(1);
        %subplot(6,1,k)
        %plot(1:N, s_True(k,:), 'g', 1:N, x_ukf(k,:), 'r')
        error_ukf(k,:) = s_True(k,:)- x_ukf(k,:);
    end
     %% RMS Error UKF %%
    for k=1:6
        x_rmsError_ukf(k, m) = rms(error_ukf(k,:));
    end
    %% Extended Kalman Filter
    t=0;
    ekf = extendedKalmanFilter(f,h,s_True(:,1),'HasAdditiveProcessNoise', true, 'ProcessNoise',Q,'HasAdditiveMeasurementNoise',true,'MeasurementNoise',R);
    x_ekf(:,1) = ekf.State;
    for k=1:N
        if k~=1
            t=t+dt;
        end
        predict(ekf,t);
        x_ekf(:,k)=correct(ekf,zV(:,k),t);
    end
    for k=1:6
        %figure(2);
        %subplot(6,1,k)
        %plot(1:N, s_True(k,:), 'g', 1:N, x_ekf(k,:), 'r')
        error_ekf(k,:) = s_True(k,:)-x_ekf(k,:);
    end
    %% RMS Error EKF %%
    for k=1:6
        x_rmsError_ekf(k, m) = rms(error_ekf(k,:));
    end
end
%% Plot %%
for k=1:6
    figure(1);
    subplot(6,1,k);
    plot(tspan, s_True(k,:), 'g', tspan, x_ekf(k,:), 'r');
    figure(2);
    subplot(6,1,k);
    plot(tspan, s_True(k,:), 'g', tspan, x_ukf(k,:), 'r');
    figure(3);
    subplot(6,1,k);
    plot(1:T, x_rmsError_ukf(k,:), 'g', 1:T, x_rmsError_ekf(k,:), 'r');  
end

%% Squre Root UKF defination
function [x,S] =ukf(f,x,S,h,z,Q,R,t)
L=numel(x);
m=numel(z);
alpha=1e-3;
ka=0;
beta=2;
lambda=alpha^2*(L+ka)-L;
C=L+lambda;
Wm=[lambda/C 0.5/C+zeros(1,2*L)];
Wc=Wm;
Wc(1)=Wc(1)+(1-alpha^2+beta);
c=sqrt(C);
X=sigmas(x,S,c);
[x1,X1,Sx,X2]=ut(f,X,Wm,Wc,L,Q,t);
%X1=sigmas(x1,P1,c);                %sigma points around x1
[z1,Z1,Sy,Z2]=ut(h,X1,Wm,Wc,m,R,t);
Pxy=X2*diag(Wc)*Z2';
K=Pxy/Sy/Sy';
x=x1+K*(z-z1);
%S=cholupdate(Sx,K*Pxy,'-');
U=K*Sy';
for i = 1:m
    Sx = cholupdate(Sx, U(:,i));
end
S=Sx;
end

%% Unscented Transform
function [y,Y,S,Y1]=ut(f,X,Wm,Wc,n,R,t)
L=size(X,2);
y=zeros(n,1);
Y=zeros(n,L);

for k=1:L
    Y(:,k)=f(X(:,k),t);
    y=y+Wm(k)*Y(:,k);
end

Y1=Y-y(:,ones(1,L));
residual=Y1*diag(sqrt(abs(Wc)));
[~,S]=qr([residual(:,2:L) R]',0);

S=cholupdate(S,residual(:,1));                %It is also right(plural)
%P=Y1*diag(Wc)*Y1'+R;  

%if Wc(1)<0
   % S=cholupdate(S,residual(:,1));
%else
    %S=cholupdate(S,residual(:,1),'+');
end

%end

%% Sigma point calculation
function X=sigmas(x,S,c)
A=c*S';
Y=x(:,ones(1,numel(x)));
X=[x Y+A Y-A];
end