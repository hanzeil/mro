%%----------trzeba opdaliæ wczeœniej pkt 2ab

%pcm
colors = zeros(length(te_labels),3);
for i=1:length(te_labels)
   if(te_labels(i)==1)
       if(te_labels(i)==roct_pcm(i,1))
           colors(i,:)=[0 1 0];
       else
           colors(i,:)=[0 0 1];
       end
   else
       if(te_labels(i)==roct_pcm(i,1))
           colors(i,:)=[1 0 0];
       else
           colors(i,:)=[0 0 0];
       end
   end
end


%pcm_all_set=te_set*pc(:,1:3);

figure
scatter(pcm_te_set(:,1),pcm_te_set(:,2),10,colors);
title('PCM (2D) - wynik klasyfikacji AdaBoost')

figure
scatter3(pcm_te_set(:,1),pcm_te_set(:,2),pcm_te_set(:,3),10,colors);
title('PCM (3D) - wynik klasyfikacji AdaBoost')

%mds

for i=1:length(te_labels)
   if(te_labels(i)==1)
       if(te_labels(i)==roct_mds(i,1))
           colors(i,:)=[0 1 0];
       else
           colors(i,:)=[0 0 1];
       end
   else
       if(te_labels(i)==roct_mds(i,1))
           colors(i,:)=[1 0 0];
       else
           colors(i,:)=[0 0 0];
       end
   end
end


figure

scatter(mds_te_set(:,1),mds_te_set(:,2),10,colors)
title('MDS (2D) - wynik klasyfikacji AdaBoost')

figure 

scatter3(mds_te_set(:,1),mds_te_set(:,2),mds_te_set(:,3),10,colors)
title('MDS (3D) - wynik klasyfikacji AdaBoost')

