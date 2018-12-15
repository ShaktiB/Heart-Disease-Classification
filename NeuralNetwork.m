function [J,w] = NeuralNetwork(Eta ,Theta, MaxNoOfIteration, Problem, Data)

%Initialization

if Problem == 1
    
    % Random initial weight vectors
    wih1 = [0.69 0.39 0.41]; %weight vector input to hidden unit no.1.
    wih2 = [0.65 0.83 0.37]; %weight vector input to hidden unit no.2.
    who1 = [0.42 0.59 0.56]; %weight vector hidden to output unit. 
    
    % Add data to feature 1,2 and label vectors.
    x1 = [-1 -1 1 1];
    x2 = [-1 1 -1 1];
    t = [-1 1 1 -1];
else
    wih1 = [0.69 0.39 0.41]; %weight vector input to hidden unit no.1.
    wih2 = [0.65 0.83 0.37]; %weight vector input to hidden unit no.2.
    who1 = [0.42 0.59 0.56]; %weight vector hidden to output unit.

    % Add data to feature 1,2 and label vectors.
    x1 = Data(:,1)';
    x2 = Data(:,2)';
    t = Data(:,3)';
end


%% Initialize number of iteration and cost.
r = 0;
J = zeros(MaxNoOfIteration,1);

%% Algorithm 

while(1)
    
    r = r + 1; %Incrementing the epoch 
    
    % Initialize gradients of the three weight vectors.
    
    DeltaWih1 = [0 0 0]; % Inputs of bias, x1,x2 to hidden neuron 1.
    DeltaWih2 = [0 0 0]; % Inputs of bias, x1,x2 to hidden neuron 2.
    DeltaWho1 = [0 0 0]; % Inputs of bias, y1,y2 to output neuron.
    
    % Initialize training sample order and predicted output.
    
    m = 0;
    Z = zeros(1,length(x1));
    
    while(m<length(x1))
        
        m = m+1;
        
        Xm = [1 x1(m) x2(m)];
        
        y1 = (wih1)*transpose(Xm);
        y2 = (wih2)*transpose(Xm);
        
        Ym = [1 tanh(y1) tanh(y2)]; % a=b=1 where sigmoid = a*tanh(b*x);
        netk_1 = Ym*transpose(who1);
        Z(m) = tanh(netk_1); 
        
        % Calculate the sensitivity value of each hidden neuron and the output neuron.
        
        DeltaO1 = (t(m)- Z(m))*(1 - ((1*tanh(1*netk_1)))^2 ); % Sensitivity value of the output neuron.
        DeltaH1 = (1 - ((1*tanh(1*y1)))^2)*who1(2)*DeltaO1; % Sensitivity value of hidden neuron 1.
        DeltaH2 = (1 - ((1*tanh(1*y2)))^2)*who1(3)*DeltaO1; % Sensitivity value of hidden neuron 2.     
        
        % Update the gradient.
        DeltaWih1 = DeltaWih1 + Eta*DeltaH1*Xm;
        DeltaWih2 = DeltaWih2 + Eta*DeltaH2*Xm;
        DeltaWho1 = DeltaWho1 + Eta*DeltaO1*Ym;  
        
    end 
    
    % Update the weight vectors.
    wih1 = wih1 + DeltaWih1; % Weight vector input to hidden unit no.1
    wih2 = wih2 + DeltaWih2; % Weight vector input to hidden unit no.2
    who1 = who1 + DeltaWho1; % Weight vector hidden to output unit.
    
    % Check the condition to stop.
     J(r) = 0.5*sqrt(sum((t-Z).^2)); % Sum of squared error
     
    if ((J(r) < Theta) || (r == MaxNoOfIteration))
        break;
    end
end 

wih1

wih2

who1

w = [wih1; wih2; who1]  

%% Testing 

i = 0;

result = zeros(1,length(x1));
y1arr = zeros(1,length(x1));
y2arr = zeros(1,length(x2));
right = 0;
wrong = 0;

while(i<length(x1))

    i = i+1;

    test_arr = [1 x1(i) x2(i)];

    test_y1 = (wih1)*transpose(test_arr);
    test_y2 = (wih2)*transpose(test_arr);
    
    y1arr(i) = test_y1;
    y2arr(i) = test_y2;

    test_Ym = [1 (1*tanh(1*test_y1)) (1*tanh(1*test_y2))]; % a=b=1 where sigmoid = a*tanh(b*x);
    test_netk_1 = test_Ym*transpose(who1); 
    test_Zm = (1*tanh(1*test_netk_1)); 
    
    result(i) = test_Zm;
end 

for i = 1:length(x1)
    if (result(i) > 0) & (t(i)>0)
        right = right+1;
    elseif (result(i) < 0) & (t(i) < 0)
        right = right+1;
    else 
        wrong = wrong+1;
    end 
end

accuracy = (right/length(x1))*100;

fprintf('The classification in percentage is: %d \n',accuracy);

%% Plotting 

% Plot learning curve 
figure;
plot(J);
xlabel('Epochs');
ylabel('J');
title('Learning Curve');

% Create dummy data for plotting wihin range of the feature values
theX1 = min(x1):0.1:max(x1);
theX2 = min(x2):0.1:max(x2);

% Obtain coordinates for the data using meshgrid
[X,Y] = meshgrid(theX1,theX2);
xx = X(:);  
yy = Y(:);

% Put the data through the feed forward function to obtain classification
zzz = zeros(length(xx),1);
y1_arr = zeros(length(xx),1);
y2_arr = zeros(length(xx),1);

for i=1:length(xx)
    
    for j=1:length(xx)
    
        test_arr = [1 xx(i) yy(i)];

        test_y1 = (wih1)*transpose(test_arr);
        test_y2 = (wih2)*transpose(test_arr);

        y1_arr(i) = tanh(test_y1);
        y2_arr(i) = tanh(test_y2);

        test_Ym = [1 (1*tanh(1*test_y1)) (1*tanh(1*test_y2))];
        test_netk_1 = test_Ym*transpose(who1); % a=b=1 
        test_Zm = (1*tanh(1*test_netk_1)); 
        
        if test_Zm > 0
            final = 1;
        end 
        if test_Zm < 0
            final = -1;
        end
        
        zzz(i) = final;
        
    end
end

% Plotting X1-X2 Space 
figure;
gscatter(xx,yy,zzz,'rgb')
xlabel('X1');
ylabel('X2');
title('X1-X2 Feature Space');

%Plotting Y1-Y2 Space
figure;
gscatter(y1_arr,y2_arr,zzz,'rgb')
xlabel('Y1');
ylabel('Y2');
title('Y1-Y2 Feature Space');

%Plotting 3-D Feature Space
C = repmat([0 0 0],numel(xx),1);

for i=1:length(xx)
    if zzz(i)> 0 
        C(i,:) = [0 1 0]; % Green if label = 1 
    else
        C(i,:) = [1 0 0]; % Red is label is = -1
    end
end

figure;
scatter3(xx,yy,zzz,[],C);
xlabel('X1');
ylabel('X2');
zlabel('N');
title('3D Classification Plot');

%Plotting 3-D Y Space 
C = repmat([0 0 0],numel(y1_arr),1);

for i=1:length(y1_arr)
    if zzz(i)> 0 
        C(i,:) = [0 1 0]; % Green if label = 1 
    else
        C(i,:) = [1 0 0]; % Red is label is = -1
    end
end

figure;
scatter3(y1_arr,y2_arr,zzz,[],C);
xlabel('Y1');
ylabel('Y2');
zlabel('N');
title('3D Y-Space Plot');

end 