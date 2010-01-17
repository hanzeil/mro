colors = zeros(length(tr_labels),3);
for i=1:length(tr_labels)
   if(tr_labels(i)==1)
       colors(i,:)=[0 1 0];
   else
       colors(i,:)=[1 0 0];
   end
end


scatter(tr_set(:,1),tr_set(:,2),10,colors)
title('Dane pocz¹tkowe (2D) - prawid³owy rozk³ad klas')

figure
scatter(pcm_tr_set(:,1),pcm_tr_set(:,2),10,colors)
title('PCM (2D) - prawid³owy rozk³ad klas')

figure
scatter3(pcm_tr_set(:,1),pcm_tr_set(:,2),pcm_tr_set(:,3),10,colors)
title('PCM (3D) - prawid³owy rozk³ad klas')