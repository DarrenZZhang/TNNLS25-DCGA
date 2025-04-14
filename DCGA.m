function [U, A, Z, obj,time,iter] = DCGA(X, ind_0, n, k, m, param)
num_view = length(X);
for i = 1:num_view
    A{i} = zeros(size(X{i},1), m*k);
end
alpha = param.alpha;
beta  = param.beta;
d = param.d;

if size(X{1},2)~=n
    for iv = 1:num_view
        X{iv} = X{iv}';
    end
end
for iv = 1:num_view
    X{iv} = NormalizeFea(X{iv},0);
end
clear iv

dims = zeros(1,num_view);
for iv = 1:num_view
    dims(iv) = size(X{iv},1);
end

Z = rand(k*m,n);

missingindex = cell(1,num_view);
for iv = 1:num_view
    temp_index = ones(1,n);
    temp_index(1,ind_0{iv}) = 0;
    missingindex{iv} = temp_index;
end

%% initialization
% initialize Y and Y_bar
I = eye(m*k); 
Y= cell(1,num_view);
for iv = 1:num_view
    Y{iv} = rand(m*m*k,m*k);
end
Y_bar = [];
for ik = 1:m*k
    Y_bar = [Y_bar repmat(I(:,ik),1,m)];
end
Y_bar = Y_bar';
clear I

% initialize R
R = cell(1,num_view);
for iv = 1:num_view
    R{iv} = zeros(dims(iv),d);
    R{iv}(1:d,1:d)=eye(d);
end

% initialize P and P_bar (H and H_bar)
temp = 0;
for iv = 1:num_view
    temp = temp +  R{iv}'*A{iv}*Y{iv}';
end
[Up,~,Sp] = svd(temp, 'econ');
P_bar = Up*Sp';
clear Up Sp temp
for iv = 1:num_view
    [Up,~,Sp] = svd(A{iv}*Y_bar', 'econ');
    P{iv} = Up*Sp';
end
clear Up Sp

% initialize A
for iv = 1:num_view
    [Ua,~,Sa] = svd(X{iv}.*(repmat(missingindex{iv},dims(iv),1))*Z'+alpha*R{iv}*P_bar*Y{iv}, 'econ');
    A{iv} = Ua*Sa';
end

w = ones(1, num_view);%/num_view;
gamma = 1;
if isfield(param, 'gamma')
    gamma = param.gamma;
end

%% Training
% MAX_ITER = 15;
MAX_ITER = 8;
tic;
for iter = 1:MAX_ITER
    
    % R step
    N = cell(1,num_view);
    for iv = 1:num_view
        N{iv} = (1+gamma) * A{iv} - P{iv}*Y_bar;
        [Ur,~,Sr] = svd(N{iv}*Y{iv}'*P_bar','econ');
        R{iv} = Ur * Sr';
    end
    clear N Ur Sr

    % Z step
    tZ1 = 0; tZ2=  0; tZ1_2 = 0;
    for iv = 1:num_view
        tZ1_2 = tZ1_2 + w(iv)^2.*(repmat(missingindex{iv},m*k,1));
        tZ2 = tZ2 + w(iv)^2*A{iv}'*(X{iv}.*(repmat(missingindex{iv},dims(iv),1)));
    end
     Z = tZ2./(tZ1_2 + beta * ones(m*k,n));
     clear tZ1 tZ2

    % A step
    B = cell(1,num_view);
    for iv = 1:num_view
        B{iv} = (P{iv}*Y_bar + gamma * R{iv} * P_bar * Y{iv})/(1+gamma);
        [Ua,~,Sa] = svd(X{iv}.*(repmat(missingindex{iv},dims(iv),1))*Z'+alpha*B{iv}, 'econ');
        A{iv} = Ua*Sa';
    end
    clear B
 
    % P step
    C = cell(1,num_view);
    for iv = 1:num_view
        C{iv} = (1+gamma)*A{iv}-gamma * R{iv}*P_bar*Y{iv};
        [Up,~,Sp] = svd(C{iv}*Y_bar','econ');
        P{iv} = Up*Sp';
    end
    clear C Up Sp
    
    
    % P_bar step
    temp = 0;
    D = cell(1,num_view);
    for iv = 1:num_view
        D{iv} = (1+gamma)*A{iv}-P{iv}*Y_bar; 
        temp = temp + R{iv}'*D{iv}*Y{iv}';
    end
    [Up,~,Sp] = svd(temp, 'econ');
    P_bar = Up*Sp';
    clear Up Sp temp D
    
    % Y_bar
    M = cell(1,num_view);
    temp = 0;
    for i = 1:num_view
        M{iv} = (1+gamma)*A{iv} - gamma * R{iv} * P_bar * Y{iv};
        temp = temp + P{iv}' * M{iv};
    end
    Y_bar = temp / num_view;
    clear M temp
     
    % Y step
    N = cell(1,num_view);
    for iv = 1:num_view
        N{iv} = (1+gamma)*A{iv} - P{iv} * Y_bar;
        Y{iv} = (1/gamma) *(P_bar'*R{iv}'*N{iv});
    end
    clear N
    

    obj(iter) = calobj(X, missingindex, A, Z, P_bar, P, Y_bar, Y, R, w, num_view, alpha, beta, gamma,dims);
    if (iter>2) && ( abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-4|| iter>MAX_ITER)
        break
    end

end
time = toc;
U = Z';

end

function obj = calobj(X, missingindex, A, Z, P_bar, P, Y_bar, Y, R, w, num_view, alpha, beta, gamma,dims)
obj = 0;
for iv = 1:num_view
    err = (X{iv}-A{iv}*Z).*(repmat(missingindex{iv},dims(iv),1));
    err2 = A{iv} - (gamma/(1+gamma))*R{iv}*P_bar*Y{iv} - (1/(1+gamma))*P{iv}*Y_bar;
    obj = obj + w(iv)^2*(sum(sum(err.*err)) + alpha*sum(sum(err2.*err2)));
end
err3 = Z;
obj = obj + beta*sum(sum(err3.*err3));

end
