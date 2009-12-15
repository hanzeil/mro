function [decTrees,adaWeigths] = SemiBoost(S,train_set, labels, iterations)


samples_count = size(train_set,1);

n_l  = size(labels,1);
n_u = samples_count-n_l;


%H=zeros(samples_count,size(train_set,2));
H=zeros(n_u,1);

for t=1:iterations
    
    ind=1;
    z=zeros(n_u,1);
    p=zeros(n_u,1);
    q=zeros(n_u,1);
    C=1;
    for i=1:n_u %%unlabeled
        %%p
        p(i)=0;
        tmp=0;
        for j=1:n_l
            %delta = 0;
            if( labels(j)==1)
                %delta=1;
                tmp=tmp+S(n_l+i,j)*exp(-2*H(i));
            end
        end
        p(i)=p(i)+tmp;
        tmp=0;
        for j=1:n_u
            tmp=tmp+(S(n_l+i,n_l+j)*exp(H(j)-H(i)));
        end
        p(i)=p(i)+C*tmp;
        
        %%q
        q(i)=0;
        tmp=0;
        for j=1:n_l
            %delta = 0;
            if( labels(j)==-1)
                %delta=1;
                tmp=tmp+S(n_l+i,j)*exp(-2*H(i));
            end
        end
        q(i)=q(i)+tmp;
        tmp=0;

        for j=1:n_u
            tmp=tmp+(S(n_l+i,n_l+j)*exp(H(i)-H(j)));
        end
        q(i)=q(i)+C*tmp;
        
        
        diff = p(i)-q(i);
        if diff~=0
            z(i)=sign(diff);
            if(z(i)==0)
               z(i)=1; 
            end
            diffs(ind)=abs(diff);
            samples_tmp(ind,:)=train_set(n_l+i,:);
            labels_tmp(ind,1)=z(i);
            ind=ind+1;
        end
           
    end

    if ind==1
       continue; 
    end
    
    [~,I]=sort(diffs,'descend');
    top10pr = ceil(0.15*size(I,2));
    
    samples_t=samples_tmp(I(1:top10pr),:);
    labels_t=labels_tmp(I(1:top10pr));
    
    samples_t(top10pr+1:top10pr+n_l,:)=train_set(1:n_l,:);
    labels_t(top10pr+1:top10pr+n_l,1)=labels;
    
    decTree =classregtree(samples_t,labels_t,'method','classification');

    samples_unlabeled = train_set(n_l+1:samples_count,:);
    
    res = str2double(eval(decTree,samples_unlabeled));
    
    error_t=0;
    for i=1:n_u
        if(res(i)==-1)
           error_t=error_t+p(i); 
        end
        if(res(i)==1)
           error_t=error_t+q(i); 
        end
    end
    
    error_t=error_t/sum(p+q);
    decTrees{t}=decTree;
    
    alfa=1/4*log((1-error_t)/error_t);
    
    adaWeigths(t) = alfa;
    
    tmpRes=res;
    H=H+alfa*tmpRes;
	

end

% Normalizacja
adaWeigths=adaWeigths/sum(adaWeigths)

