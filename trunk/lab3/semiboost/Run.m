%**************************************************************************
%pkt 3 - baza - SPAM 
% TR = importdata('../data/Spam/spambase.data', ',');
% for i=1:size(TR,2)-1
%     TR(:,i)=mat2gray(TR(:,i));
% end
% pr25 = floor(0.25*size(TR,1));
% pr50 = 2*pr25;
% pr75 = 3*pr25;
% 
% tr_set = TR(1:pr75, 1:size(TR,2)-1);
% te_set = TR(pr75+1:size(TR,1), 1:size(TR,2)-1);
% tr_labels =  TR(1:pr25, size(TR,2));
% te_labels =  TR(pr75+1:size(TR,1), size(TR,2));
% 
% tr_n = size(tr_set,1);
% te_n = size(te_set,1);
%**************************************************************************

%**************************************************************************
%pkt 3 - baza - Heart
% TR = importdata('../data/Heart/heart.data', ',');
% 
% pr25 = floor(0.25*size(TR,1));
% pr50 = 2*pr25;
% pr75 = 3*pr25;
% 
% tr_set = TR(1:pr75, 2:size(TR,2));
% te_set = TR(pr75+1:size(TR,1), 2:size(TR,2));
% tr_labels =  TR(1:pr25, 1);
% te_labels =  TR(pr75+1:size(TR,1), 1);
% 
% tr_n = size(tr_set,1);
% te_n = size(te_set,1);
%**************************************************************************


%**************************************************************************
%pkt 3 - baza - balony
% TR = importdata('../data/Balony/adult+stretch.data', ',');
% 
% pr25 = floor(0.25*size(TR,1));
% pr50 = 2*pr25;
% pr75 = 3*pr25;
% 
% tr_set = TR(1:pr75, 1:size(TR,2)-1);
% te_set = TR(pr75+1:size(TR,1), 1:size(TR,2)-1);
% tr_labels =  TR(1:pr25, size(TR,2));
% te_labels =  TR(pr75+1:size(TR,1), size(TR,2));
% 
% tr_n = size(tr_set,1);
% te_n = size(te_set,1);
%**************************************************************************

%**************************************************************************
%pkt 3 - baza - Cancer 
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
tr_labels =  TR(1:pr25, 2);
te_labels =  TR(pr75+1:size(TR,1), 2);

tr_n = size(tr_set,1);
te_n = size(te_set,1);

%**************************************************************************

   
learn_iteration =10;

for j=1:size(tr_labels,1)
    if(tr_labels(j)~=1)
        tr_labels(j)=-1;
    end
end
for j=1:size(te_labels,1)
    %te_labels(j)=te_labels(j)+1;
    if(te_labels(j)~=1)
        te_labels(j)=-1;
    end
end
tr_error = zeros(1,learn_iteration);
te_error = zeros(1,learn_iteration);

S=zeros(size(tr_set,1));
for i=1:size(tr_set,1) 
    for j=1:size(tr_set,1)
        difr=tr_set(i,:)-tr_set(j,:);
        sqr=difr.^2;
        dis = -(sum(sqr));
        tmp=exp(dis);
        S(i,j)=tmp;
    end    
end


for i=1:learn_iteration
	[trees,weigths] = SemiBoost(S,tr_set,tr_labels,i);
	hits_tr = SemiBoostTest(weigths,trees,tr_set(1:pr25,:),tr_labels);
	tr_error(i) = (pr25-hits_tr)/pr25;
    hits_te=SemiBoostTest(weigths,trees,te_set,te_labels);
	te_error(i) = (te_n-hits_te)/te_n;
end


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

