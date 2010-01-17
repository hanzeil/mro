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


%pkt 1 - baza - Cancer trochê inny rozk³ad danych
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


%pkt 2c
for feat_no=1:size(tr_set,2)
    for i=1:learn_iteration
        [trees,weigths] = AdaBoost(tr_set(:,feat_no),tr_labels,i);
        [L_tr,classes,hits_tr,roct] = AdaBoostEval(weigths,trees,tr_set(:,feat_no),tr_labels);
        tr_error(feat_no,i) = (tr_n-hits_tr)/tr_n;
%         [L_te,classes,hits_te,roct] = AdaBoostEval(weigths,trees,te_set,te_labels);
%         te_error(i) = (te_n-hits_te)/te_n;

        [x y w1 auc(feat_no,i) w2 w3 w4]=perfcurve(roct(:,2),roct(:,1),1);

    end
end


plot(tr_error(:,10))
xlabel('cecha')
ylabel('b³¹d treningowy')

figure
plot(auc(:,10))
xlabel('cecha')
ylabel('pole pod ROC')

[best_feat,bfIx] = sort(auc(:,10),'descend')

cnt=1;
for featNum = [3,5,10,20]
    tr_feat_set=zeros(size(tr_set,1),featNum);
    te_feat_set=zeros(size(te_set,1),featNum);
    for i=1:featNum
        tr_feat_set(:,i)=tr_set(:,bfIx(i));
        te_feat_set(:,i)=te_set(:,bfIx(i));
    end
    for i=1:learn_iteration
            [trees,weigths] = AdaBoost(tr_feat_set,tr_labels,i);
            [L_tr,classes,hits_tr,roct] = AdaBoostEval(weigths,trees,tr_feat_set,tr_labels);
            tr_error(featNum,i) = (tr_n-hits_tr)/tr_n;
            
            [L_te,classes,hits_te,roct] = AdaBoostEval(weigths,trees,te_feat_set,te_labels);
            te_error(featNum,i) = (te_n-hits_te)/te_n;

            [x y w1 auc(featNum,i) w2 w3 w4]=perfcurve(roct(:,2),roct(:,1),1);
            
            
    end
    
    figure
    %subplot(4,1,cnt);
    plot(1:learn_iteration,te_error(featNum,:));
    axis([1,learn_iteration,0,1]);
    title(['Testing Error - ' num2str(featNum) ' best features']);
    xlabel('weak classifier number');
    ylabel('error rate');
    grid on;
    
    cnt=cnt+1;
end

figure
plot([3,5,10,20],te_error([3,5,10,20],10),'--rs');
    axis([3,20,0,0.1]);
    title(['Testing Error']);
    xlabel('liczba cech');
    ylabel('b³¹d klasyfikacji');
    grid on;

    figure
    plot([3,5,10,20],auc([3,5,10,20],10),'--rs');
    axis([3,20,0,1]);
    title(['AUROC']);
    xlabel('liczba cech');
    ylabel('Pole pod ROC');
    grid on;

