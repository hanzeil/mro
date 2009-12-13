learn_iteration =10;

M = importdata('../data/Balony/yellow-small+adult-stretch.data', ',');
%M = importdata('../data/Balony/adult+stretch.data', ',');
tr_n = size(M,1);
te_n = size(M,1);
tr_set = M(:,1:size(M,2)-1);
te_set = M(:,1:size(M,2)-1);

tr_labels =  M(:,size(M,2));
te_labels =  M(:,size(M,2));

tr_error = zeros(1,learn_iteration);
te_error = zeros(1,learn_iteration);

for i=1:learn_iteration
	[trees,weigths] = AdaBoost(tr_set,tr_labels,i);
	[L_tr,hits_tr] = AdaBoostEval(weigths,trees,tr_set,tr_labels);
	tr_error(i) = (tr_n-hits_tr)/tr_n;
	[L_te,hits_te] = AdaBoostEval(weigths,trees,te_set,te_labels);
	te_error(i) = (te_n-hits_te)/te_n;
end

%view(adaboost_model.decTrees{1})
figure;
subplot(1,2,1); 
plot(1:learn_iteration,tr_error);
axis([1,learn_iteration,0,1]);
title('Training Error');
xlabel('weak classifier number');
ylabel('error rate');
grid on;

subplot(1,2,2); axis square;
plot(1:learn_iteration,te_error);
axis([1,learn_iteration,0,1]);
title('Testing Error');
xlabel('weak classifier number');
ylabel('error rate');
grid on;

