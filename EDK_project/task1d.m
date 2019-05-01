%% Clean
close all;
clc;
clear all;

%% Get data

x_feature1=load('Iris_TTT4275/class_1','-ascii');
x_feature2=load('Iris_TTT4275/class_2','-ascii');
x_feature3=load('Iris_TTT4275/class_3','-ascii');

x_all=[x_feature1; x_feature2; x_feature3];

x_target=[1*ones(size(x_feature1,1),1);
          2*ones(size(x_feature2,1),1);
          3*ones(size(x_feature3,1),1) ];

%% Training data

% Last 30 for training
trainingSamples=30;
testSamples=size(x_feature1,1)-trainingSamples;

features=size(x_all,2);
x_training = [x_feature1(testSamples+1:trainingSamples+testSamples,1:features);
              x_feature2(testSamples+1:trainingSamples+testSamples,1:features); 
              x_feature3(testSamples+1:trainingSamples+testSamples,1:features)];
          
x_target_training=[1*ones(trainingSamples,1); 
                2*ones(trainingSamples,1);
                3*ones(trainingSamples,1)];

%% Test data
% first 20 for testing
x_testing = [x_feature1(1:testSamples,1:features); 
             x_feature2(1:testSamples,1:features); 
             x_feature3(1:testSamples,1:features)];
         
x_target_testing=[1*ones(testSamples,1); 
               2*ones(testSamples,1);
               3*ones(testSamples,1)];
%% Include offset
numberOfClasses = 3;
offset_training=ones(numberOfClasses*trainingSamples,1);
offset_testing=ones(numberOfClasses*testSamples,1);

x_training=[x_training offset_training];
x_testing=[x_testing offset_testing];


%% Learning
w_0=1;
W=zeros(numberOfClasses,features+w_0);

learningRate=0.003;

MSE_summation=0;
MSE=zeros(1000,1);
der_MSE=0;
errors=[];
confusionMatrix_training=zeros(3,3);

for u=1:1000
    
    numberOfErrors=0;
    MSE_summation=0;
    der_MSE=0;
    
    for i=1:size(x_training,1)
    
        % Hypotheses function
        z=W*x_training(i,1:features+w_0)';

        % Sigmoid function
        g=1./(1+exp(-z));

        % Define the targets
        t=zeros(numberOfClasses,1); %3x1
        t(x_target_training(i))=1;

        % Derivative of MSE

        der_MSE= der_MSE + ((g-t).*g.*(1-g))*x_training(i,1:features+w_0);

        % Calculate MSE
        MSE_summation= MSE_summation + (g-t)'*(g-t);
        
        % Calculate numberOfErrors
        
        [~,decison] = max(g);
        if(decison ~= x_target_training(i))
            numberOfErrors=numberOfErrors+1;
        end
        
        % Calculate confusion matrix
        
        if u == 1000
            [~,decison] = max(g);
            confusionMatrix_training(x_target_training(i),decison) = confusionMatrix_training(x_target_training(i), decison)+1;
        end
        

    end
    MSE(u)=0.5*MSE_summation;
  
    errors_in_training(u)=numberOfErrors;
    estimatedErrors_average=numberOfErrors/90;
    errors(u)=estimatedErrors_average; 

    W=W-(learningRate*der_MSE);
end


%% error and confusion matrix

all_errors=[errors' errors_in_training'];

% plots
figure(1);
t=[0:1:1000-1];
plot(t,MSE)

figure(2);
plot(t,all_errors(:,1))

figure(3);
plot(t,all_errors(:,2))

% Find accurary of training
accurary=sum(diag(confusionMatrix_training))/size(x_training,1);

% Misclassification rate of training set
error_rate=1-accurary;

disp(['Error rate for training set ', num2str(error_rate*100), ' %'])


%% Test the test set

z_test=W*x_testing';
g_test=1./(1+exp(-z_test));

% Decision rule
[~,decison]=max(g_test);

% Error and confusion matrix in test set
confusionMatrix_testing=zeros(3,3);
error=0;
for i=1:size(x_testing,1)
    confusionMatrix_testing(x_target_testing(i),decison(i))=confusionMatrix_testing(x_target_testing(i),decison(i))+1;
    if(decison(i) ~= x_target_testing(i) )
        error=error+1;
    end
end

% Find accurary of testing
accurary_testing=sum(diag(confusionMatrix_testing))/size(x_testing,1);

% Misclassification rate of testing set
error_rate_testing=1-accurary_testing;

disp(['Error rate for training set ', num2str(error_rate_testing*100), ' %'])


