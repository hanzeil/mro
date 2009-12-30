TR = importdata('../data/Cancer/cancer_TR75.data', ',');
TE = importdata('../data/Cancer/cancer_TE25.data', ',');
tr_n = size(TR,1);
te_n = size(TE,1);
tr_set = TR(:,3:size(TR,2));
te_set = TE(:,3:size(TE,2));
tr_labels =  TR(:,2);
te_labels =  TE(:,2);


[pc,score,latent,tsquare] = princomp(TR)