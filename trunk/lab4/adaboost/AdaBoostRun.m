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
%dzia�a ok, od 1 iteracji 0 b��d�w dla TR i TE
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


TR = importdata('../data/Spam/spambase.data', ',');
for i=1:size(TR,2)-1
    TR(:,i)=mat2gray(TR(:,i));
end
pr25 = floor(0.25*size(TR,1));
pr50 = 2*pr25;
pr75 = 3*pr25;

tr_set = TR(1:pr75, 1:size(TR,2)-1);
te_set = TR(pr75+1:size(TR,1), 1:size(TR,2)-1);
tr_labels =  TR(1:pr75, size(TR,2));
te_labels =  TR(pr75+1:size(TR,1), size(TR,2));

tr_n = size(tr_set,1);
te_n = size(te_set,1);

%dzia�a ok, od 3 iteracji 0 b��d�w dla TR i od 1 dla TE
%**************************************************************************

%**************************************************************************
%pkt 4 - baza - Heart
% TR = importdata('../data/Heart/heart_TR25.data', ',');
% TE = importdata('../data/Heart/heart_TE25.data', ',');
% tr_n = size(TR,1);
% te_n = size(TE,1);
% tr_set = TR(:,2:size(TR,2));
% te_set = TE(:,2:size(TE,2));
% tr_labels =  TR(:,1);
% te_labels =  TE(:,1);
%dzia�a ok
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
%dzia�a ok
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
%dzia�a ok, TR po 3 oteracjach 0 b��d�w, TE w okolicach 0,05
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
%dzia�a ok, TR po 3 oteracjach 0 b��d�w, TE w okolicach 0,1
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
	[trees,weigths] = AdaBoost(tr_set,tr_labels,i);
	[L_tr,hits_tr,roct] = AdaBoostEval(weigths,trees,tr_set,tr_labels);
	tr_error(i) = (tr_n-hits_tr)/tr_n;
	[L_te,hits_te,roct] = AdaBoostEval(weigths,trees,te_set,te_labels);

	te_error(i) = (te_n-hits_te)/te_n;
    [x y]=perfcurve(roct(:,2),roct(:,1),1);
    X(i)=x(2); 
    Y(i)=y(2);
end

% [tpr,fpr,thresholds]=roc(roct(:,2),roct(:,1));
% figure;
%posi=ones(size(roct(:,1)));
%[X,Y]=perfcurve(roct2(:,:,2),roct2(:,:,1),1);

figure;
[X2 ind]=sort(X);
%X2=X;
X3(1)=0;
Y3(1)=0;
X3(2:size(X2,2)+1)=X2;
X3(size(X2,2)+2)=1;

Y2=Y(ind);
%Y2=Y;
Y3(2:size(Y2,2)+1)=Y2;
Y3(size(Y2,2)+2)=1;
plot(X3,Y3);

P=0;
for i=2:size(X3,2)
    P=P+(X3(i)-X3(i-1))*((Y3(i)+Y3(i-1))/2);
end
P=P-0.5;
P

% plot(1:fpr,tpr);
% %axis([1,learn_iteration,0,1]);
% title('Training Error');
% xlabel('fpr');
% ylabel('tpf');
% grid on;

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

