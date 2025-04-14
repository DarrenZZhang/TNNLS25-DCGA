function [folds] = gengerate_folder_1(n,num_view,mr)
    rng(1);
    num_iter = 10;
    folds = cell(1,num_iter);
    num_miss = floor(n*mr);
    for iv = 1:num_iter
        fold = ones(n,num_view);
        miss_idx = randperm(n,num_miss);
        for i = num_miss
            miss_view = randperm(num_view,1);
            fold(miss_idx(i),miss_view)=0;
        end
        folds{iv} = fold;
    end
end

