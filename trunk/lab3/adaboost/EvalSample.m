function [L,hits,error_rate] = EvalSample(decTree,test_set,sample_weights,true_labels)


ind = str2double(eval(decTree,test_set));

hits = sum(ind==true_labels);
error_rate = sum(sample_weights(ind~=true_labels));

classes=length(unique(true_labels));
if (classes == 1) classes = 2; end
L = zeros(length(ind),classes);

for i=1:classes
    L(ind==i,i) = 1;
end
%L(ind==1,1) = 1;
%L(ind==2,2) = 1;
% ind
% L
% hits
% error_rate

