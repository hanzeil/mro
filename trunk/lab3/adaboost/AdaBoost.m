function [decTrees,adaWeigths] = AdaBoost(train_set, labels, iterations)

adaboost_model = struct('weights',zeros(1,iterations),...
						'decTrees',[]); %cell(1,iterations));

samples_count = size(train_set,1);
weigths = ones(samples_count,1)/samples_count;

for t=1:iterations
    decTrees{t} = classregtree(train_set,labels,'method','classification','weights',weigths);

	[L,hits,error_t] = EvalSample(decTrees{t},train_set,weigths,labels);
	if(error_t==1)
		error_t=1-eps;
	elseif(error_t==0)
		error_t=eps;
    end
    
    %parametr beta
    beta = error_t/(1-error_t);
	adaWeigths(t) = log10(1/beta);
    
    %L
	%C=likelihood2class(L);
    C=zeros(size(train_set,1),1);
    for i=1:size(train_set,1)
       %znalezienie dla ktorej klasy mamy niezerow¹ wartoœæ
       C(i) = find(L(i,:),1);
    end
    %poprawnie sklasyfikowane
	correct_labels=(C==labels);
    %L
    %correct_labels
	% dla poprawnie sklasyfikowanych przemna¿amy przez wspó³czynnik
	weigths(correct_labels) = weigths(correct_labels)*beta;					

	% Normalizacja
	weigths = weigths/sum(weigths);
end

% Normalizacja
adaWeigths=adaWeigths/sum(adaWeigths);

