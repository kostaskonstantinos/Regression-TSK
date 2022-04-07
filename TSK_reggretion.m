
%% ANFIS EXAMPLE
format compact
clear all
clc

%% Load data - Split data
data=load('airfoil_self_noise.dat');
preproc=1;
[trnData,chkData,tstData]=split_data(data,preproc);

%% Evaluation function
Rsq = @(ypred,y) 1-sum((y-ypred).^2)/sum((y-mean(y)).^2);

%% FIS with grid partition
n = input('Enter the number of model : ');
switch (n)
    case 1
        fis=genfis1(trnData,2,'gbellmf','constant');
    case 2
         fis=genfis1(trnData,3,'gbellmf','constant');
    case 3
        fis=genfis1(trnData,2,'gbellmf','linear');
    case 4
        fis=genfis1(trnData,3,'gbellmf','linear');
    otherwise
        disp('other value')
end
% before training
figure
[trnFis,trnError,~,valFis,valError]=anfis(trnData,fis,100,[],chkData);
 for j = 1:prod(size(valFis.Inputs))
        subplot(5,1,j)
        plotmf(fis, 'input', j);
        title(['Input ', num2str(j)])
 end
 Y=sgtitle(['Initial Inputs Of Model ',num2str(n)])
 %plotmf(fis,'input',size(trnData,2)-1);

%% No Validation
figure
plot(trnError,'LineWidth',2); grid on;
legend('Training Error');
xlabel('# of Iterations'); ylabel('Error');
title('ANFIS Hybrid Training - No Validation');
figure
 for j = 1:prod(size(valFis.Inputs))
        subplot(5,1,j)
        % before
         plotmf(valFis, 'input', j);
         title(['input',num2str(j)] )
 end
 Y=sgtitle(['Trained Inputs Of Model ',num2str(n)])

%% Validation
figure
plot([trnError valError],'LineWidth',2); grid on;
xlabel('# of Iterations'); ylabel('Error');
legend('Training Error','Validation Error');
title(['Learning Curve Of Model ',num2str(n)]);

%////////////////////////////////////
Yh=evalfis(valFis,tstData(:,1:end-1));
Y=tstData(:,end);
error=Y-Yh;
R2=Rsq(Yh,tstData(:,end));
RMSE=sqrt(mse(Yh,tstData(:,end)));
nmse=1-R2;
ndei = sqrt(nmse);
figure
   plot(error);xlabel('check Samples'); ylabel('Error')
   title(['Prediction Error Of Model ',num2str(n)])

