function [L,hits,roctab] = AdaBoostEval(adaWeigths,decTrees,test_set,...
                                true_labels)


hypothesis_n = length(adaWeigths);
sample_n = size(test_set,1);
class_n = length(unique(true_labels));
if (class_n == 1) class_n = 2; end
temp_L = zeros(sample_n,class_n,hypothesis_n);

% dla ka¿dej iteracji przydzielamy klasy dla poszczególnych przyk³adów
for i=1:hypothesis_n
	[temp_L(:,:,i),hits,error_rate] = EvalSample(decTrees{i},...
													 test_set,ones(sample_n,1),true_labels);
    %temp_L(:,:,i)
	temp_L(:,:,i) = temp_L(:,:,i)*adaWeigths(i);
    %temp_L(:,:,i)
end

L = sum(temp_L,3);
%L
%likelihood2class(L)

[sample_n,class_n] = size(L);
%dyskretyzacja do 0-1
maxs = (L==repmat(max(L,[],2),[1,class_n]));
classes=zeros(sample_n,1);
for i=1:sample_n
    %znalezienie dla ktorej klasy mamy niezerow¹ wartoœæ
	classes(i) = find(maxs(i,:),1);
end

%classes
roctab(:,1)=classes;
roctab(:,2)=classes==true_labels;

hits = sum(classes==true_labels);

