%% Clean

clear all;
close all;
clc;

%% Data and feature extraction
% i..1 - 3 (classes) , j..1 - 4 (features)

% Classes

%Class 1 - Iris setosa (w_1)
%Class 2 - Iris-versicolor(w_2)
%Class 3 - Iris-virginica(w_3)

x_w_1_all=load('Iris_TTT4275/class_1','-ascii');
x_w_2_all=load('Iris_TTT4275/class_2','-ascii');
x_w_3_all=load('Iris_TTT4275/class_3','-ascii');

x_feature_names=["sepal length","sepal width","petal length","petal width"];
x_target_names=["setosa","versicolor","virginica"];

x_features=[x_w_1_all; x_w_2_all; x_w_3_all;];
        

%% Parameters and constants
TrainingData=30;
TestData=20;
NumberOfClasses=size(x_target_names,2);
NumberOfFeatures=size(x_features,2);
ObservationsOfEachClass=size(x_features,1);

%% Training data

x_training=NaN;

for i=1:NumberOfFeatures
    for j=1:TrainingData*NumberOfClasses
        x_training(j,i)=x_features(j,i);
    end
end

x_target_training=NaN;

for i=1:NumberOfClasses
    for j=1:TrainingData
        x_target_training(j,i)=(i-1)*ones(1,1);
    end
end

x_target_training = [x_target_training(:,1); x_target_training(:,2); x_target_training(:,3)];


%% Test Data

x_testing=NaN;

for i=1:NumberOfFeatures
    for j=1:TestData*NumberOfClasses
        x_testing(j,i)=x_features(j,i);
    end
end

x_target_testing=NaN;

for i=1:NumberOfClasses
    for j=1:TestData
        x_target_testing(j,i)=(i-1)*ones(1,1);
    end
end

x_target_testing=[x_target_testing(:,1); x_target_testing(:,2); x_target_testing(:,3)];

%% Offset w_0

offset=1;

%% Training and testing data

x_all_Training = [x_training ones(TrainingData*NumberOfClasses,1) x_target_training];
x_all_Testing = [x_testing ones(TestData*NumberOfClasses,1) x_target_testing];


%% Constants

W=zeros(NumberOfClasses,NumberOfFeatures+offset);

stepFactor=0.01;

nabla_W_MSE=0;
nabla_gk_MSE=NaN;
nabla_zk_g=NaN;
nabla_W_zk=NaN;
g_k=NaN;
z_k=NaN;
t_k=NaN;

%% Sigmoid function

for iteration=1:3
    
    nabla_W_MSE=0;
    number_Of_errors=0;
    
    for i=1:size(x_all_Training,1)
    
        % Define decresent function z=W*x
        z_k = W*x_all_Training(i,1:size(x_training,2)+offset)';

        % Define sigmoid
        g_k =(1./(1+exp(-z_k)));

        % Define target
        t_k=zeros(NumberOfClasses,1);
        t_k(x_all_Training(i,size(x_all_Training,2))+1)=1.0

        % Minimum square error gradient technique
        nabla_gk_MSE = g_k-t_k;
        nabla_zk_g = g_k.*(1-g_k);
        nabla_W_zk = x_all_Training(i,1:size(x_training,2)+offset);

        nabla_W_MSE = nabla_W_MSE + ((nabla_gk_MSE.*nabla_zk_g)*nabla_W_zk)
        
        % Calculate decision
        [~,decision]=max(z_k);
        
        if decision ~= x_all_Training(i,size(x_all_Training,2))+1
            number_Of_errors += 1;
        end
        

        
        
        

    end
    
    % Move the gradient in the oppsite direction
    W=W-(stepFactor*nabla_W_MSE);
    
    
    
    
    
end












