all_set = zeros(size(tr_set,1)+size(te_set,1),size(tr_set,2));

all_labels = zeros(size(tr_labels,1)+size(te_labels,1),size(tr_labels,2));

all_set(1:size(tr_set,1),:)=tr_set;
all_set(size(tr_set,1)+1:size(all_set,1),:)=te_set;


all_labels(1:size(tr_labels,1),:)=tr_labels;
all_labels(size(tr_labels,1)+1:size(all_labels,1),:)=te_labels;
%pkt 3d


[pc,z,latent,tsquare] = princomp(all_set);

pcm_all_set=all_set*pc(:,1:3);

%for met={'ward','complete'}
    Y = pdist(pcm_all_set); 
    Z = linkage(Y,'complete'); 
    T = cluster(Z,'maxclust',2);
    hits = sum(T==all_labels)
    percentage=hits/length(T)
%end

colors = zeros(length(all_labels),3);
for i=1:length(all_labels)
   if(all_labels(i)==1)
       colors(i,:)=[0 1 0];
   else
       colors(i,:)=[1 0 0];
   end
end



scatter(pcm_all_set(:,1),pcm_all_set(:,2),10,colors)
title('PCM (2D)-wynik klasteryzacji')

figure

scatter3(pcm_all_set(:,1),pcm_all_set(:,2),pcm_all_set(:,3),10,colors)
title('PCM (3D)-wynik klasteryzacji')


%mds
dissimilarities = pdist(all_set);
YY=mdscale(dissimilarities,3);

    Y = pdist(YY); 
    Z = linkage(Y,'complete'); 
    T = cluster(Z,'maxclust',2);
    hits = sum(T==all_labels)
    percentage=hits/length(T)
%end

colors = zeros(length(all_labels),3);
for i=1:length(all_labels)
   if(all_labels(i)==1)
       colors(i,:)=[0 1 0];
   else
       colors(i,:)=[1 0 0];
   end
end

figure

scatter(YY(:,1),YY(:,2),10,colors)
title('MDS (2D)-wynik klasteryzacji')

figure 
scatter3(YY(:,1),YY(:,2),YY(:,3),10,colors)
title('MDS (3D)-wynik klasteryzacji')