%% Clean
close all;
clc;
clear all;

%% Get data

x_feature1=load('Iris_TTT4275/class_1','-ascii');
x_feature2=load('Iris_TTT4275/class_2','-ascii');
x_feature3=load('Iris_TTT4275/class_3','-ascii');

%% Histograms

% feature 1 - class 1 to 3
figure(1);
subplot(3,1,1)
%Class 1 feature 1
histogram(x_feature1(:,1),15)
xlim([0 8])
title('Feature 1')
% Class 2 feature 1
subplot(3,1,2)
histogram(x_feature2(:,1),15)
xlim([0 8])
subplot(3,1,3)
% Class 3 feature 1
histogram(x_feature3(:,1),15)
xlim([0 8])

figure(2);
subplot(3,1,1)
%Class 1 feature 2
histogram(x_feature1(:,2),15)
xlim([0 8])
title('Feature 2')
% Class 2 feature 2
subplot(3,1,2)
histogram(x_feature2(:,2),15)
xlim([0 8])
subplot(3,1,3)
% Class 3 feature 2
histogram(x_feature3(:,2),15)
xlim([0 8])


figure(3);
subplot(3,1,1)
%Class 1 feature 3
histogram(x_feature1(:,3),15)
xlim([0 8])
title('Feature 3')
% Class 2 feature 3
subplot(3,1,2)
histogram(x_feature2(:,3),15)
xlim([0 8])
subplot(3,1,3)
% Class 3 feature 3
histogram(x_feature3(:,3),15)
xlim([0 8])


figure(4);
subplot(3,1,1)
%Class 1 feature 4
histogram(x_feature1(:,4),15)
xlim([0 8])
title('Feature 4')
% Class 2 feature 4
subplot(3,1,2)
histogram(x_feature2(:,4),15)
xlim([0 8])
subplot(3,1,3)
% Class 3 feature 4
histogram(x_feature3(:,4),15)
xlim([0 8])

% Correlation of each feature

figure(5)
hold on
subplot(2,2,1)
histogram([x_feature1(:,1),x_feature2(:,1), x_feature3(:,1)],15)
xlim([0 8])
title('Correlation of feature 1')
subplot(2,2,2)
histogram([x_feature1(:,2),x_feature2(:,2), x_feature3(:,2)],15)
xlim([0 8])
subplot(2,2,3)
histogram([x_feature1(:,3),x_feature2(:,3), x_feature3(:,3)],15)
xlim([0 8])
subplot(2,2,4)
histogram([x_feature1(:,4),x_feature2(:,4), x_feature3(:,4)],15)
xlim([0 8])

%% Training and testing data


% Choose number of features and the use of last 30 or first 30 samples in the training set

numberOffeatures=4;
useLastTraining=1;

if numberOffeatures == 4
        x_all=[x_feature1; x_feature2; x_feature3];

        x_target=[1*ones(size(x_feature1,1),1);
                  2*ones(size(x_feature2,1),1);
                  3*ones(size(x_feature3,1),1) ];
         
              
elseif numberOffeatures == 3
        disp('Second feature deleted')
        x_feature1(:,2)=[];
        x_feature2(:,2)=[];
        x_feature3(:,2)=[];

        x_all=[x_feature1; x_feature2; x_feature3];

        x_target=[1*ones(size(x_feature1,1),1);
                  2*ones(size(x_feature2,1),1);
                  3*ones(size(x_feature3,1),1) ];
      
elseif numberOffeatures == 2
        disp('First and second features deleted')
        x_feature1(:,1:2)=[];
        x_feature2(:,1:2)=[];
        x_feature3(:,1:2)=[];

        x_all=[x_feature1; x_feature2; x_feature3];

        x_target=[1*ones(size(x_feature1,1),1);
                  2*ones(size(x_feature2,1),1);
                  3*ones(size(x_feature3,1),1) ];
   
elseif numberOffeatures == 1
        disp('First, second and third features deleted')
        x_feature1(:,1:3)=[];
        x_feature2(:,1:3)=[];
        x_feature3(:,1:3)=[];

        x_all=[x_feature1; x_feature2; x_feature3];

        x_target=[1*ones(size(x_feature1,1),1);
                  2*ones(size(x_feature2,1),1);
                  3*ones(size(x_feature3,1),1) ]; 
end

%% Choose last or first for training set and test set

if useLastTraining == 1
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
                
    % First 20 for testing
    x_testing = [x_feature1(1:testSamples,1:features); 
                 x_feature2(1:testSamples,1:features); 
                 x_feature3(1:testSamples,1:features)];

    x_target_testing=[1*ones(testSamples,1); 
                   2*ones(testSamples,1);
                   3*ones(testSamples,1)];        
elseif useLastTraining == 0
    
    % First 30 for training
    trainingSamples=30;
    features=size(x_all,2);
    testSamples=size(x_feature1,1)-trainingSamples;
    numberOfClasses = 3;

    x_training = [x_feature1(1:trainingSamples,1:features);
                  x_feature2(1:trainingSamples,1:features); 
                  x_feature3(1:trainingSamples,1:features)];

    x_target_training=[1*ones(trainingSamples,1); 
                    2*ones(trainingSamples,1);
                    3*ones(trainingSamples,1)];
                
    % Last 20 for testing
    x_testing = [x_feature1(trainingSamples+1:trainingSamples+testSamples,1:features); 
                 x_feature2(trainingSamples+1:trainingSamples+testSamples,1:features); 
                 x_feature3(trainingSamples+1:trainingSamples+testSamples,1:features)];

    x_target_testing=[1*ones(testSamples,1); 
                   2*ones(testSamples,1);
                   3*ones(testSamples,1)];             
end
      
%% Include offset

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
confusionMatrix_training=zeros(numberOfClasses,numberOfClasses);

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
    estimatedErrors_average=numberOfErrors/size(x_training,1);
    errors(u)=estimatedErrors_average; 

    W=W-(learningRate*der_MSE);
end


%% Error and confusion matrix

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
disp('Confusion matrix for training_set')
disp(confusionMatrix_training)

%% Test the test set

z_test=W*x_testing';
g_test=1./(1+exp(-z_test));

% Decision rule
[~,decison]=max(g_test);

% Error and confusion matrix in test set
confusionMatrix_testing=zeros(numberOfClasses,numberOfClasses);
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

disp(['Error rate for testing set ', num2str(error_rate_testing*100), ' %'])
disp('Confusion matrix for testing_set')
disp(confusionMatrix_testing)



