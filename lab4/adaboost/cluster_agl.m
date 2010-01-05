%**************************************************************************
% TE = importdata('../data/Heart/heart_TE25.data', ',');
% TE = TE(:,2:size(TE,2));

% TE = importdata('../data/Spam/spambase_TE25.data', ',');
% TE = TE(:,1:size(TE,2)-1);

TE = importdata('../data/Cancer/cancer_TR75.data', ',');


te_labels =  TE(:,2);
TE = TE(:,3:size(TE,2));

for j=1:size(TE,1)
    te_labels(j)=te_labels(j)+1;
end
%pkt 3a
size(TE(:,1))
for met={'ward','complete'}
    Y = pdist(TE); 
    Z = linkage(Y,met); 
    T = cluster(Z,'maxclust',2);
    hits = sum(T==te_labels)
    percentage=hits/length(T)
end