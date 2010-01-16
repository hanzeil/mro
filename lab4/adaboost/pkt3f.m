%%----------trzeba opdaliæ wczeœniej pkt 2ab
colors = zeros(length(te_labels),3);
for i=1:length(te_labels)
   if(te_labels(i)==1)
       if(te_labels(i)==roct(i,1))
           colors(i,:)=[0 1 0];
       else
           colors(i,:)=[0 0 1];
       end
   else
       if(te_labels(i)==roct(i,1))
           colors(i,:)=[1 0 0];
       else
           colors(i,:)=[0 0 0];
       end
   end
end

Y = pdist(te_set); 
Z = linkage(Y,'complete'); 
T = cluster(Z,'maxclust',2);
hits = sum(T==te_labels)
percentage=hits/length(T)

% for i=length(tr_labels)+1:length(all_labels)
%    if(all_labels(i)==1)
%        colors(i,:)=[0 0 1];
%    else
%        colors(i,:)=[1 1 0];
%    end
% end



scatter(te_set(:,1),te_set(:,2),10,colors)
title('Dane pocz¹tkowe (2D)')


%pcm
[pc,z,latent,tsquare] = princomp(te_set);

pcm_all_set=te_set*pc(:,1:3);

figure
scatter(pcm_all_set(:,1),pcm_all_set(:,2),10,colors);
title('PCM (2D)')

figure
scatter3(pcm_all_set(:,1),pcm_all_set(:,2),pcm_all_set(:,3),10,colors);
title('PCM (3D)')

%mds
dissimilarities = pdist(te_set);
Y=mdscale(dissimilarities,3);

figure
title('MDS (2D)')
scatter(Y(:,1),Y(:,2),10,colors)


figure 
title('MDS (3D)')
scatter3(Y(:,1),Y(:,2),Y(:,3),10,colors)

