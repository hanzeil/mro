all_set = zeros(size(tr_set,1)+size(te_set,1),size(tr_set,2));

all_labels = zeros(size(tr_labels,1)+size(te_labels,1),size(tr_labels,2));

all_set(1:size(tr_set,1),:)=tr_set;
all_set(size(tr_set,1)+1:size(all_set,1),:)=te_set;


all_labels(1:size(tr_labels,1),:)=tr_labels;
all_labels(size(tr_labels,1)+1:size(all_labels,1),:)=te_labels;


colors = zeros(length(all_labels),3);
for i=1:length(tr_labels)
   if(all_labels(i)==1)
       colors(i,:)=[0 1 0];
   else
       colors(i,:)=[1 0 0];
   end
end

for i=length(tr_labels)+1:length(all_labels)
   if(all_labels(i)==1)
       colors(i,:)=[0 0 1];
   else
       colors(i,:)=[0 0 0];
   end
   
end



scatter(all_set(:,1),all_set(:,2),10,colors) %ew gplotmatrix
title('Dane pocz¹tkowe (2D)')


%pcm
[pc,z,latent,tsquare] = princomp(all_set);

pcm_all_set=all_set*pc(:,1:3);

figure
scatter(pcm_all_set(:,1),pcm_all_set(:,2),10,colors);
title('PCM (2D)')

figure
scatter3(pcm_all_set(:,1),pcm_all_set(:,2),pcm_all_set(:,3),10,colors);
title('PCM (3D)')

%mds
dissimilarities = pdist(all_set);
Y=mdscale(dissimilarities,3);

figure

scatter(Y(:,1),Y(:,2),10,colors)
title('MDS (2D)')


figure 

scatter3(Y(:,1),Y(:,2),Y(:,3),10,colors)
title('MDS (3D)')

