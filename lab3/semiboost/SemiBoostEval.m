function result = SemiBoostEval(adaWeigths,decTrees,samples)

hypothesis_n = length(adaWeigths);

sum=zeros(size(samples,1),1);

for i=1:hypothesis_n
	res = str2double(eval(decTrees{i},samples));
    sum=sum+res*adaWeigths(i);
end

result = sign(sum);

