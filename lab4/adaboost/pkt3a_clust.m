all_set = zeros(size(tr_set,1)+size(te_set,1),size(tr_set,2));

all_labels = zeros(size(tr_labels,1)+size(te_labels,1),size(tr_labels,2));

all_set(1:size(tr_set,1),:)=tr_set;
all_set(size(tr_set,1)+1:size(all_set,1),:)=te_set;


all_labels(1:size(tr_labels,1),:)=tr_labels;
all_labels(size(tr_labels,1)+1:size(all_labels,1),:)=te_labels;
%pkt 3a

%for met={'ward','complete'}
    Y = pdist(TE); 
    Z = linkage(Y,'complete'); 
    T = cluster(Z,'maxclust',2);
    hits = sum(T==TE_L)
    percentage=hits/length(T)
%end