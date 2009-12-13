%
% DEMONSTRATION OF ADABOOST_tr and ADABOOST_te
%
% Just type "demo" to run the demo.
%
% Using adaboost with linear threshold classifier 
% for a two class classification problem.
%
% Bug Reporting: Please contact the author for bug reporting and comments.
%
% Cuneyt Mertayak
% email: cuneyt.mertayak@gmail.com
% version: 1.0
% date: 21/05/2007

weak_learner_n =10;

M = importdata('../data/Balony/yellow-small+adult-stretch.data', ',');
%M = importdata('../data/Balony/adult+stretch.data', ',');
tr_n = size(M,1);
te_n = size(M,1);
tr_set = M(:,1:4);
te_set = M(:,1:4);

tr_labels =  M(:,5);
te_labels =  M(:,5);

tr_error = zeros(1,weak_learner_n);
te_error = zeros(1,weak_learner_n);

for i=1:weak_learner_n
	[trees,weigths] = AdaBoost(tr_set,tr_labels,i);
	[L_tr,hits_tr] = AdaBoostEval(weigths,trees,tr_set,tr_labels);
	tr_error(i) = (tr_n-hits_tr)/tr_n;
	[L_te,hits_te] = AdaBoostEval(weigths,trees,te_set,te_labels);
	te_error(i) = (te_n-hits_te)/te_n;
end

%view(adaboost_model.decTrees{1})
figure;
subplot(1,2,1); 
plot(1:weak_learner_n,tr_error);
axis([1,weak_learner_n,0,1]);
title('Training Error');
xlabel('weak classifier number');
ylabel('error rate');
grid on;

subplot(1,2,2); axis square;
plot(1:weak_learner_n,te_error);
axis([1,weak_learner_n,0,1]);
title('Testing Error');
xlabel('weak classifier number');
ylabel('error rate');
grid on;

