%**************************************************************************
%baza - Cancer
%pkt 1
% TR = importdata('../data/Cancer/cancer_TR75.data', ',');
% TE = importdata('../data/Cancer/cancer_TE25.data', ',');
% tr_n = size(TR,1);
% te_n = size(TE,1);
% tr_set = TR(:,3:size(TR,2));
% te_set = TE(:,3:size(TE,2));
% tr_labels =  TR(:,2);
% te_labels =  TE(:,2);


%pkt 1 - baza - Cancer troch� inny rozk�ad danych
TR = importdata('../data/Cancer/cancer.data', ',');

for i=3:size(TR,2)
    TR(:,i)=mat2gray(TR(:,i));
end

pr25 = floor(0.25*size(TR,1));
pr50 = 2*pr25;
pr75 = 3*pr25;

tr_set = TR(1:pr75, 3:size(TR,2));
tr_set=mat2gray(tr_set);
te_set = TR(pr75+1:size(TR,1), 3:size(TR,2));
te_set=mat2gray(te_set);
tr_labels =  TR(1:pr75, 2);
te_labels =  TR(pr75+1:size(TR,1), 2);

tr_n = size(tr_set,1);
te_n = size(te_set,1);

learn_iteration =10;

for j=1:size(tr_labels,1)
    tr_labels(j)=tr_labels(j)+1;
end
for j=1:size(te_labels,1)
    te_labels(j)=te_labels(j)+1;
end
tr_error = zeros(1,learn_iteration);
te_error = zeros(1,learn_iteration);
pcm_tr_error = zeros(1,learn_iteration);
pcm_te_error = zeros(1,learn_iteration);
mds_tr_error = zeros(1,learn_iteration);
mds_te_error = zeros(1,learn_iteration);

%pkt 2a
[pc,z,latent,tsquare] = princomp(tr_set);

pcm_tr_set=tr_set*pc(:,1:5);
pcm_te_set=te_set*pc(:,1:5);

dissimilarities = pdist(TR(:,3:size(TR,2)));
mdsY=mdscale(dissimilarities,3);
mds_tr_set=mdsY(1:pr75, :);
mds_te_set=mdsY(pr75+1:size(TR,1), :);

%pkt 2b
for i=1:learn_iteration
	[trees,weigths] = AdaBoost(tr_set,tr_labels,i);
	[L_tr,classes,hits_tr,roct] = AdaBoostEval(weigths,trees,tr_set,tr_labels);
	tr_error(i) = (tr_n-hits_tr)/tr_n;
	[L_te,classes,hits_te,roct] = AdaBoostEval(weigths,trees,te_set,te_labels);
    te_error(i) = (te_n-hits_te)/te_n;
    
    [x y w1 auc(i) w2 w3 w4]=perfcurve(roct(:,2),roct(:,1),1);

    X(i)=x(2); 
    Y(i)=y(2);
    %pcm
    
    [trees,weigths] = AdaBoost(pcm_tr_set,tr_labels,i);
	[L_tr,classes,hits_tr,roct] = AdaBoostEval(weigths,trees,pcm_tr_set,tr_labels);
    pcm_tr_error(i) = (tr_n-hits_tr)/tr_n;
	
    [L_te,classes,hits_te,roct_pcm] = AdaBoostEval(weigths,trees,pcm_te_set,te_labels);
    pcm_te_error(i) = (te_n-hits_te)/te_n;
    
    [x y w1 auc_pcm(i) w2 w3 w4] = perfcurve(roct_pcm(:,2),roct_pcm(:,1),1);
    
    %mds
    
    [trees,weigths] = AdaBoost(mds_tr_set,tr_labels,i);
	[L_tr,classes,hits_tr,roct] = AdaBoostEval(weigths,trees,mds_tr_set,tr_labels);
    mds_tr_error(i) = (tr_n-hits_tr)/tr_n;
	
    [L_te,classes,hits_te,roct_mds] = AdaBoostEval(weigths,trees,mds_te_set,te_labels);
    mds_te_error(i) = (te_n-hits_te)/te_n;
    
    [x y w1 auc_mds(i) w2 w3 w4] = perfcurve(roct_mds(:,2),roct_mds(:,1),1);
    
end

err=te_error(learn_iteration)
err_pcm=pcm_te_error(learn_iteration)
err_mds=mds_te_error(learn_iteration)
auroc=auc(learn_iteration)
auroc_pcm=auc_pcm(learn_iteration)
auroc_mds=auc_pcm(learn_iteration)

% [tpr,fpr,thresholds]=roc(roct(:,2),roct(:,1));
% figure;
%posi=ones(size(roct(:,1)));
%[X,Y]=perfcurve(roct2(:,:,2),roct2(:,:,1),1);

% figure;
% [X2 ind]=sort(X);
% %X2=X;
% X3(1)=0;
% Y3(1)=0;
% X3(2:size(X2,2)+1)=X2;
% X3(size(X2,2)+2)=1;
% 
% Y2=Y(ind);
% %Y2=Y;
% Y3(2:size(Y2,2)+1)=Y2;
% Y3(size(Y2,2)+2)=1;
% plot(X3,Y3);
% 
% P=0;
% for i=2:size(X3,2)
%     P=P+(X3(i)-X3(i-1))*((Y3(i)+Y3(i-1))/2);
% end
% P=P-0.5;
% P
% P2=SampleError(classes,te_labels,'AUROC')
% plot(1:fpr,tpr);
% %axis([1,learn_iteration,0,1]);
% title('Training Error');
% xlabel('fpr');
% ylabel('tpf');
% grid on;

%view(adaboost_model.decTrees{1})
figure;
subplot(3,2,1); 
plot(1:learn_iteration,tr_error);
axis([1,learn_iteration,0,1]);
title('Training Error - original data');
xlabel('weak classifier number');
ylabel('error rate');
grid on;

subplot(3,2,2); axis square;
plot(1:learn_iteration,te_error);
axis([1,learn_iteration,0,1]);
title('Testing Error - original data');
xlabel('weak classifier number');
ylabel('error rate');
grid on;

subplot(3,2,3); 
plot(1:learn_iteration,pcm_tr_error);
axis([1,learn_iteration,0,1]);
title('Training Error - top 5 PCM features');
xlabel('weak classifier number');
ylabel('error rate');
grid on;

subplot(3,2,4); axis square;
plot(1:learn_iteration,pcm_te_error);
axis([1,learn_iteration,0,1]);
title('Testing Error - top 5 PCM features');
xlabel('weak classifier number');
ylabel('error rate');
grid on;

subplot(3,2,5); 
plot(1:learn_iteration,mds_tr_error);
axis([1,learn_iteration,0,1]);
title('Training Error - MDS');
xlabel('weak classifier number');
ylabel('error rate');
grid on;

subplot(3,2,6); axis square;
plot(1:learn_iteration,mds_te_error);
axis([1,learn_iteration,0,1]);
title('Testing Error - MDS');
xlabel('weak classifier number');
ylabel('error rate');
grid on;

