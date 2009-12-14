%**************************************************************************
%pkt 4 - baza - SPAM
% TR = importdata('../data/Spam/spambase_TR25.data', ',');
% TE = importdata('../data/Spam/spambase_TE25.data', ',');
% tr_n = size(TR,1);
% te_n = size(TE,1);
% tr_set = TR(:,1:size(TR,2)-1);
% te_set = TE(:,1:size(TE,2)-1);
% tr_labels =  TR(:,size(TR,2));
% te_labels =  TE(:,size(TE,2));
%dzia³a ok, od 1 iteracji 0 b³êdów dla TR i TE
%**************************************************************************

%**************************************************************************
%pkt 5 - baza - SPAM
% TR = importdata('../data/Spam/spambase_TR75.data', ',');
% TE = importdata('../data/Spam/spambase_TE25.data', ',');
% tr_n = size(TR,1);
% te_n = size(TE,1);
% tr_set = TR(:,1:size(TR,2)-1);
% te_set = TE(:,1:size(TE,2)-1);
% tr_labels =  TR(:,size(TR,2));
% te_labels =  TE(:,size(TE,2));
% %dzia³a ok, od 3 iteracji 0 b³êdów dla TR i od 1 dla TE
%**************************************************************************

%**************************************************************************
%pkt 4 - baza - Heart
% TR = importdata('../data/Heart/heart_TR50.data', ',');
% TE = importdata('../data/Heart/heart_TE25.data', ',');
% tr_n = size(TR,1);
% te_n = size(TE,1);
% tr_set = TR(:,2:size(TR,2));
% te_set = TE(:,2:size(TE,2));
% tr_labels =  [];%TR(:,1);
% te_labels =  TE(:,1);
%dzia³a ok
%**************************************************************************

%**************************************************************************
%pkt 5 - baza - Heart
% TR = importdata('../data/Heart/heart_TR75.data', ',');
% TE = importdata('../data/Heart/heart_TE25.data', ',');
% tr_n = size(TR,1);
% te_n = size(TE,1);
% tr_set = TR(:,2:size(TR,2));
% te_set = TE(:,2:size(TE,2));
% tr_labels =  TR(:,1);
% te_labels =  TE(:,1);
%dzia³a ok
%**************************************************************************

%**************************************************************************
%pkt 4 - baza - Cancer
% TR = importdata('../data/Cancer/cancer_TR25.data', ',');
% TE = importdata('../data/Cancer/cancer_TE25.data', ',');
% tr_n = size(TR,1);
% te_n = size(TE,1);
% tr_set = TR(:,3:size(TR,2));
% te_set = TE(:,3:size(TE,2));
% tr_labels =  TR(:,2);
% te_labels =  TE(:,2);
%dzia³a ok, TR po 3 oteracjach 0 b³êdów, TE w okolicach 0,05
%**************************************************************************

%**************************************************************************
%pkt 5 - baza - Cancer
% TR = importdata('../data/Cancer/cancer_TR75.data', ',');
% TE = importdata('../data/Cancer/cancer_TE25.data', ',');
% tr_n = size(TR,1);
% te_n = size(TE,1);
% tr_set = TR(:,3:size(TR,2));
% te_set = TE(:,3:size(TE,2));
% tr_labels =  TR(:,2);
% te_labels =  TE(:,2);
%dzia³a ok, TR po 3 oteracjach 0 b³êdów, TE w okolicach 0,1
%**************************************************************************


%**************************************************************************
%pkt 4 - baza - balony
TR = importdata('../data/Balony/adult+stretch_TR75.data', ',');
TE = importdata('../data/Balony/adult+stretch_TE25.data', ',');
tr_n = size(TR,1);
te_n = size(TE,1);
tr_set = TR(:,1:size(TR,2)-1);
te_set = TE(:,1:size(TE,2)-1);
tr_labels =  [];%TR(:,size(TR,2));
te_labels =  TE(:,size(TE,2));
%dzia³a ok
%**************************************************************************



%M = importdata('../data/Balony/yellow-small+adult-stretch.data', ',');
%M = importdata('../data/Spam/spambase.data', ',');
%M = importdata('../data/Ozon/onehr.data', ',');
%TR = importdata('../data/Heart/heart.data', ',');
%TE = importdata('../data/Heart/heart.data', ',');
%M = importdata('../data/Balony/adult+stretch.data', ',');
% TR = M;
% TE = M;
% tr_n = size(TR,1);
% te_n = size(TE,1);
% tr_set = TR(:,1:size(TR,2)-1);
% te_set = TE(:,1:size(TE,2)-1);
% 
% tr_labels =  TR(:,size(TR,2));
% te_labels =  TE(:,size(TE,2));

learn_iteration =10;

for j=1:size(tr_labels,1)
    tr_labels(j)=tr_labels(j)+1;
end
for j=1:size(te_labels,1)
    te_labels(j)=te_labels(j)+1;
end
tr_error = zeros(1,learn_iteration);
te_error = zeros(1,learn_iteration);

for i=1:learn_iteration
	[trees,weigths] = SemiBoost(tr_set,tr_labels,i);
% 	[L_tr,hits_tr] = SemiBoostEval(weigths,trees,tr_set,tr_labels);
% 	tr_error(i) = (tr_n-hits_tr)/tr_n;
	[L_te,hits_te] = SemiBoostEval(weigths,trees,te_set,te_labels);
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

