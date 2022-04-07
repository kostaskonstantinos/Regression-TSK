
%% ANFIS EXAMPLE
% format compact
% clear all
% clc

%% Load data - Split data
data=csvread('train.csv',1,0);
preproc=1;
[trnData,chkData,tstData]=split_data(data,preproc);

%% Evaluation function
Rsq = @(ypred,y) 1-sum((y-ypred).^2)/sum((y-mean(y)).^2);

Nfeatures= [3 6 10 15 25 ];
rad = [0.3 0.4 0.5 0.6 0.8];

% create a table to find the min(error)
%bestModel =zeros( length( Nfeatures ) ,length( rad ));

% k-fold cross validation 
k=5
n= length(trnData(:,end));
c = cvpartition(n,'KFold',k);
   % Feature selection neigh=10
[idx,weights] = relieff( trnData(:,1:end-1), trnData(:,end),10);

output=(trnData(:,end));

            
% for i = 1:numel(Nfeatures)
%     i
%     for j = 1:numel(rad)
%         for kei = 1:c.NumTestSets
%               trnPart =find( c.training(kei)==1);              
%               trnFinal = [trnData(trnPart,idx(1:Nfeatures(i))) output(trnPart) ];
%               chkPart=find( c.test(kei)==1);    
%               chkFinal = [trnData(chkPart,idx(1:Nfeatures(i))) output(chkPart)];
% 
%               fis=genfis2(trnFinal(:,1:end-1),trnFinal(:,end),rad(j));
%              [trainFis,trainError,~,valFis,valError] = anfis(trnFinal,fis,[100],[],chkFinal);
%             bestModel(i,j)=bestModel(i,j)+mean(valError);
%            
%         end
%     end
% end
% bestModel = bestModel ./ 5

%optional choice
mError = min(bestModel(:));
[row,col] = find(bestModel==mError);

%plot optional choice
figure;
stem3(rad,Nfeatures,bestModel,'-o','Color','r','MarkerSize',10,'MarkerFaceColor','#D9FFFF')
xlabel('radious')
ylabel('numberOfFeatures')
zlabel('mean error')
title('grid search')
 hold on
 stem3(rad(col),Nfeatures(row),mError,'-o','Color','g','MarkerSize',10,'MarkerFaceColor','g')
 
 %print the features used
 featuresUsed=idx(1:Nfeatures(row))
 
 %final model
 finalNfeatures=Nfeatures(row);
 finalrad=rad(col);
 trnFinal = [trnData(:,idx(1:finalNfeatures)) trnData(:,end) ];
 chkFinal = [chkData(:,idx(1:finalNfeatures)) chkData(:,end) ];
 tstFinal = [tstData(:,idx(1:finalNfeatures)) tstData(:,end) ];
 fis=genfis2(trnFinal(:,1:end-1),trnFinal(:,end),finalrad);
 [trainFis,trainError,~,valFis,valError] = anfis(trnFinal,fis,[100],[],chkFinal);
            
 %plot membership func before and after training of final model
figure
count=1;
%1-10
for j = 1:10
        subplot(5,4,count)
        count=count+1;
        plotmf(fis, 'input', j);   
        title(['input before',num2str(j)] )
        subplot(5,4,count)
        count=count+1;
        plotmf(valFis, 'input', j);
        title(['input after',num2str(j)] )
end
 
 figure 
 count=1;
 %11-20
  for j = 11:20
        subplot(5,4,count)
        count=count+1;
        plotmf(fis, 'input', j);   
        title(['input before',num2str(j)] )
        subplot(5,4,count)
        count=count+1;
        plotmf(valFis, 'input', j);
        title(['input after',num2str(j)] )
  end
  
  figure
  count=1
  %21-25
  for j = 21:25
        subplot(5,2,count)
        count=count+1;
        plotmf(fis, 'input', j);   
        title(['input before',num2str(j)] )
        subplot(5,2,count)
        count=count+1;
        plotmf(valFis, 'input', j);
        title(['input after',num2str(j)] )
  end
 
 % plot Learning Curve
figure
plot([trainError valError],'LineWidth',2); grid on;
xlabel('# of Iterations'); ylabel('Error');
legend('Training Error','Validation Error');
title('Learning Curve Of Final Model ');

Yh=evalfis(valFis,tstFinal(:,1:end-1));
Y=tstFinal(:,end);
error=Y-Yh;
R2=Rsq(Yh,tstFinal(:,end));
RMSE=sqrt(mse(Yh,tstFinal(:,end)));
nmse=1-R2;
ndei = sqrt(nmse);
% prediction error of final model
figure
plot(error);xlabel('check Samples'); ylabel('Error')
title('Prediction Error Of Final Model ')
% Y,Yh
plot([Y(1:500) Yh(1:500)])
xlabel('200 samples'); 
ylabel('Output'); 
title('predictions')
