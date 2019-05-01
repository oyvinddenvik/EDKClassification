close all;
clear all;
clc;

x1all=load('Iris_TTT4275/class_1','-ascii');
x2all=load('Iris_TTT4275/class_2','-ascii');
x3all=load('Iris_TTT4275/class_3','-ascii');

%g = W*x + w_0;
%W := [W w_0] (4x4)
W = zeros(3,5); %initializing weight matrix as zeroes

%inserting one column of ones for the offset, and one column with the 
%target class
x1all = [x1all ones(50,1) ones(50,1)];
x2all = [x2all ones(50,1) 2*ones(50,1)];
x3all = [x3all ones(50,1) 3*ones(50,1)];

%vector of training and test samples
x_train = [x1all(1:30,:) ; x2all(1:30,:) ; x3all(1:30,:)];
x_test = [x1all(31:50,:) ; x2all(31:50,:) ; x3all(31:50,:)];
error_train = []; %vector of training error for each iteration
a = 0.001; %learning rate
i = 1;

%training with gradient descent
for t = 1:3
    %initializing the gradient and the training error
    dMSE = 0;
    error = 0;
    
    %iterate through the training samples
    for k = 1:90
        z_k = W*x_train(k,1:5)'
        g_k = zeros(3,1);
        t_k = zeros(3,1);
        t_k(x_train(k,6)) = 1;
        for j = 1:3
            g_k(j) = 1/(1+exp(-z_k(j)));
        end
        %calculate gradient
        dMSE = dMSE + ((g_k-t_k).*g_k.*(1-g_k))*x_train(k,1:5);
        
        %calculate training errors
        [~,I] = max(W*x_train(k,1:5)');
        if I ~= x_train(k,6)
            error = error +1;
        end
        %disp(I)
        
    end
    W = W - a*dMSE;
    %disp(error)
    %calculate average training error for each iteration
    error_train = [error_train error/90];
    
    %stop if validation error increades for five consecutive iterations
    if i >= 100 && sum(error_train(i) > error_train(i-5:i-1)) == 5
        break
    end
    i = i + 1;
end