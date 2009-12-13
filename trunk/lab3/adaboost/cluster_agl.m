M = importdata('../data/Balony/yellow-small+adult-stretch.data', ',');
for met={'ward','complete'}
    Y = pdist(M); 
    Z = linkage(Y,met); 
    T = cluster(Z,'maxclust',2);
    met
    T
end