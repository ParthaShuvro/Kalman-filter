%% Monte Carlo %%
n = 4;
T = 100;
x_rmsError_ukf = zeros(n,T);
x_rmsError_ekf = zeros(n,T);
%rng(10);
for m=1:100   
    %% Initialization %%
    %n = 4;
    dt = 15;     % seconds
    G = 6.6742e-11;
    M = 5.98e24;
    Q = diag([0, 10e-6, 0, 0]);
    R = diag([100^2, 0.1^2]);
    f=@(x)[x(1)+x(2)*dt;x(2)+((x(1)*(x(4))^2-(G*M)/(x(1))^2))*dt;x(3)+ x(4)*dt;x(4)+((-2*x(2)*x(4))/x(1))*dt];
    h=@(x)[x(1); x(3)];
    s=[6571000; 7900; 0; 0.0012];
    x=s+sqrt(Q)*randn(4,1);
    S=zeros(n);
    simTime = 3600; % seconds
    tspan = 0:dt:simTime;
    N = length(tspan);
    x_ukf = zeros(n,N);
    x_ekf = zeros(n,N);
    s_True = zeros(n,N);
    error_ukf = zeros(n,N);
    error_ekf = zeros(n,N);
    zV = zeros(2,N);

    %% Unscented Kalman Filter %%
    for k=1:N
        z = h(s) + sqrt(R)*randn(2,1);
        s_True(:,k) = s;
        zV(:,k) = z;
        [x, S] = ukf(f,x,S,h,z,Q,R);
        x_ukf(:,k) = x;
        s = f(s) + sqrt(Q)*randn(4,1);
    end
    for k=1:4  
        %figure(1);
        %subplot(4,1,k)
        %plot(1:N, s_True(k,:), 'g', 1:N, x_ukf(k,:), 'r')
        error_ukf(k,:) = s_True(k,:)- x_ukf(k,:); 
    end

    %% RMS Error UKF %%
    for k=1:4
        x_rmsError_ukf(k, m) = rms(error_ukf(k,:));
    end

    %% Extended Kalman Filter %%

    ekf = extendedKalmanFilter(f,h,s_True(:,1),'HasAdditiveProcessNoise', true, 'ProcessNoise',Q,'HasAdditiveMeasurementNoise',true,'MeasurementNoise',R);
    x_ekf(:,1) = ekf.State;
    for k=1:N
        predict(ekf);
        x_ekf(:,k)=correct(ekf,zV(:,k));
    end
    for k=1:4
        %figure(2);
        %subplot(4,1,k)
        %plot(1:N, s_True(k,:), 'g', 1:N, x_ekf(k,:), 'r')
        error_ekf(k,:) = s_True(k,:)-x_ekf(k,:);
    end
    %% RMS Error EKF %%
    for k=1:4
        x_rmsError_ekf(k, m) = rms(error_ekf(k,:));
    end
end

%% Plot %%
for k=1:4
    figure(1);
    subplot(4,1,k);
    plot((1:N)*15, s_True(k,:), 'g', (1:N)*15, x_ekf(k,:), 'r');
    figure(2);
    subplot(4,1,k);
    plot((1:N)*15, s_True(k,:), 'g', (1:N)*15, x_ukf(k,:), 'r');   
    %legend('True','Estimated');
    figure(3);
    subplot(4,1,k);
    plot(1:T, x_rmsError_ukf(k,:), 'g', 1:T, x_rmsError_ekf(k,:), 'r');
    %title('Monte Carlo Error'),
    %legend('UKF Error','EKF Error');    
end

%% Squre Root UKF defination %%
function [x,S] =ukf(f,x,S,h,z,Q,R)
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
[x1,X1,Sx,X2]=ut(f,X,Wm,Wc,L,Q);
%X1=sigmas(x1,P1,c);                %sigma points around x1
[z1,Z1,Sy,Z2]=ut(h,X1,Wm,Wc,m,R);
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
function [y,Y,S,Y1]=ut(f,X,Wm,Wc,n,R)
L=size(X,2);
y=zeros(n,1);
Y=zeros(n,L);

for k=1:L
    Y(:,k)=f(X(:,k));
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