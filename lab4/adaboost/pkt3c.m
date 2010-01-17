colors = zeros(length(tr_labels),3);
for i=1:length(tr_labels)
   if(tr_labels(i)==1)
       colors(i,:)=[0 1 0];
   else
       colors(i,:)=[1 0 0];
   end
end


dissimilarities = pdist(pcm_tr_set);
Y=mdscale(dissimilarities,2);


scatter(Y(:,1),Y(:,2),10,colors)
title('MDS (2D) - prawid�owy rozk�ad klas')


% distances = pdist(Y);
% plot(dissimilarities,distances,'bo', ...
% [0 max(dissimilarities)],[0 max(dissimilarities)],'k:');
% xlabel('Dissimilarities'); ylabel('Distances')
% 



figure 

Y=mdscale(dissimilarities,3);
scatter3(Y(:,1),Y(:,2),Y(:,3),10,colors)
title('MDS (3D) - prawid�owy rozk�ad klas')