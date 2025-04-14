clear;
clc;
warning off;
addpath(genpath('./'));
rng(2024)

%% dataset
ds = {'BBCSport'};
metric = {'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'};
i_fold = 1;
model = 'DCGA';

for dsi = 1:length(ds)
    %% load data & make folder
    dataName = ds{dsi};
    load(dataName);
    k = length(unique(Y));
    n = length(Y);
    num_view = length(X);
    original_X = X;
    original_Y = Y;
    
    for i_miss = 1:1
        X = original_X;
        Y = original_Y;
        mr = i_miss * 0.1;
        foldName = strcat(dataName,'_percentDel_',num2str(mr),'.mat');
        load(foldName);
        filename = strcat(dataName,'-incomplete-',model,'-mr-',num2str(mr),'.txt');
        ind_folds = folds{i_fold};
        %% construct W
        ind_0 = cell(1,num_view);
        ind_1 = cell(1,num_view);
        for iv = 1:num_view            
            ind_0{iv} = find(ind_folds(:,iv) == 0);
            ind_1{iv} = find(ind_folds(:,iv) == 1);
        end
        %% initialize
        for iv = 1:length(X)
            if size(X{iv},2)~= n
                X{iv} = X{iv}';
            end
            X{iv} = NormalizeFea(X{iv}, 0);
            X{iv}(:,ind_0{iv}) = 0;
        end
        %% param setting
        MM = [3];
        AP = [1e-3, 1e-2, 1e-1, 1e0, 1e1];
        BT = [1e-3, 1e-2, 1e-1, 1e0, 1e1];
        %% 
        perf = []; perf_svd=[];
        f = fopen(filename,'a');
        for rp0 = 1:length(MM)
            m = MM(rp0);
            for rp1 = 1:length(AP)
                for rp2 = 1:length(BT)   
                     param.alpha = AP(rp1);
                     param.beta = BT(rp2);
                tic
                param.d = k;
                [U, A, Z, obj] = DCGA(X, ind_0, n, k, m, param);
                print_result(U,Y,k,f,m,param);
                time  = toc;  
                end
            end
        end
        fclose(f);
        clear X Y
    end
end
    
function [] = print_result(U,Y,k,f,m,param)
    [res] = myNMIACCwithmean(U,Y,k); %[ACC nmi Purity Fscore Precision Recall AR Entropy];
    fprintf("\n m: %f, alpha: %f, beta: %f, \t ACC nmi Purity Fscore Precision Recall AR Entropy: %.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n", m, param.alpha, param.beta, res(1), res(2),res(3),res(4),res(5),res(6),res(7),res(8));
    fprintf(f,"\n m: %f, alpha: %f, beta: %f, \t ACC nmi Purity Fscore Precision Recall AR Entropy: %.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f \n", m, param.alpha, param.beta, res(1), res(2),res(3),res(4),res(5),res(6),res(7),res(8));
end