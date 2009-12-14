function [decTrees,adaWeigths] = SemiBoost(train_set, labels, iterations)
%
% ADABOOST TRAINING: A META-LEARNING ALGORITHM
%  adaboost_model = ADABOOST_tr(tr_func_handle,te_func_handle,
%                               train_set,labels,iterations)
% 
%        'tr_func_handle' and 'te_func_handle' are function handles for
%        training and testing of a weak learner, respectively. The weak learner
%        has to support the learning in weighted datasets. The prototypes
%        of these functions has to be as follows.
%
%        model = train_func(train_set,sample_weights,labels)
%                    train_set: a TxD-matrix where each row is a training sample in
%                        a D dimensional feature space.
%                    sample_weights: a Tx1 dimensional vector, the i-th entry 
%                        of which denotes the weight of the i-th sample.
%                    labels: a Tx1 dimensional vector, the i-th entry of which
%                        is the label of the i-th sample.
%                    model: the output model of the training phase, which can 
%                        consists of parameters estimated.
%
%        [L,hits,error_t] = test_func(model,test_set,sample_weights,true_labels)
%                    model: the output of train_func
%                    test_set: a KxD dimensional matrix, each of whose row is a
%                        testing sample in a D dimensional feature space.
%                    sample_weights:  a Dx1 dimensional vector, the i-th entry 
%                        of which denotes the weight of the i-th sample.
%                    true_labels: a Dx1 dimensional vector, the i-th entry of which
%                        is the label of the i-th sample.
%                    L: a Dx1-array with the predicted labels of the samples.
%                    hits: number of hits, calculated with the comparison of L and
%                        true_labels.
%                    error_t: number of misses divided by the number of samples.
%        
%
%        'train_set' contains the samples for training and it is NxD matrix
%        where N is the number of samples and D is the dimension of the
%        feature space. 'labels' is an Nx1 matrix containing the class
%        labels of the samples. 'iterations' is the number of weak
%        learners to be used.
%
%        The output 'adaboost_model' is a structure with the fields 
%         - 'weights': 1x'iterations' matrix specifying the weights
%                      of the resulted weighted majority voting combination 
%         - 'parameters': 1x'iterations' structure matrix specifying
%                         the special parameters of the hypothesis that is
%                         created at the corresponding iteration of 
%                         learning algorithm
% 
%        Specific Properties That Must Be Satisfied by The Function pointed
%        by 'func_handle'
%        ------------------------------------------------------------------
%
% Note: Labels must be positive integers from 1 upto the number of classes.
% Node-2: Weighting is done as specified in AIMA book, Stuart Russell et.al. (sec edition)
%
% Bug Reporting: Please contact the author for bug reporting and comments.
%
% Cuneyt Mertayak
% email: cuneyt.mertayak@gmail.com
% version: 1.0
% date: 21/05/2007


samples_count = size(train_set,1);

n_l  = size(labels,1);
n_u = samples_count-n_l;

weigths = ones(samples_count,1)/samples_count;

for i=1:size(train_set,1) 
    for j=1:size(train_set,1)
        dis = -(sum((train_set(i,:)-train_set(j,:)).^2));
        tmp=exp(dis);
        S(i,j)=tmp;
    end    
end

S

H=zeros(n_u);

for t=1:iterations
    
    ind=1;
    z=zeros(n_u,1);
    for i=1:n_u %%unlabeled
        %%p
        p(i)=0;
        tmp=0;
        for j=1:n_l
            delta = 0;
            if( labels(j)==1)
                delta=1;
            end
            tmp=tmp+S(n_l+i,j)*exp(-2*H(i))*delta;
        end
        p(i)=p(i)+tmp;
        tmp=0;
        for j=1:n_u
            tmp=tmp+(S(n_l+i,n_l+j)*exp(H(j)-H(i)));
        end
        p(i)=p(i)+tmp;
        
        %%q
        q(i)=0;
        tmp=0;
        for j=1:n_l
            delta = 0;
            if( labels(j)==2)
                delta=1;
            end
            tmp=tmp+S(n_l+i,j)*exp(-2*H(i))*delta;
        end
        q(i)=q(i)+tmp;
        tmp=0;
        for j=1:n_u
            tmp=tmp+(S(n_l+i,n_l+j)*exp(H(i)-H(j)));
        end
        q(i)=q(i)+tmp;
        
        diff = p(i)-q(i);
        if(diff>0)
            z(i)=1; 
        else
            z(i)=2;
        end
        if(diff==0)
           vvvvv=09; 
        else
           samples_t(ind)=train_set(n_l+i);
           labels_t(ind)=z(i);
           weigths_t(ind)=weigths(n_l+i);
           ind=ind+1;
        end
           
    end
    z
    
    if(ind==1)
       continue; 
    end
    
    decTrees{t} =classregtree(train_set,z,'method','classification','weights',weigths);

	[L,hits,error_t] = EvalSample(decTrees{t},samples_t,weigths_t,labels_t);
	if(error_t==1)
		error_t=1-eps;
	elseif(error_t==0)
		error_t=eps;
    end
    
    %parametr beta
    beta = error_t/(1-error_t);
	adaWeigths(t) = log10(1/beta);
    
    
	%C=likelihood2class(L);
    C=zeros(size(samples_t,1),1);
    for i=1:size(samples_t,1)
       %znalezienie dla ktorej klasy mamy niezerow¹ wartoœæ
       C(i) = find(L(i,:),1);
    end
    %poprawnie sklasyfikowane
	correct_labels=(C==labels_t);
    %L
    correct_labels
	% dla poprawnie sklasyfikowanych przemna¿amy przez wspó³czynnik
	%weigths(correct_labels) = weigths(correct_labels)*beta;					

	% Normalizacja
	weigths = weigths/sum(weigths);
    
    weigths
end

% Normalizacja
%adaWeigths=adaWeigths/sum(adaWeigths);

