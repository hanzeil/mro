function hits = SemiBoostTest(adaWeigths,decTrees,test_set,true_labels)
        res=SemiBoostH(adaWeigths,decTrees,test_set);
        correct = true_labels==res;
        hits = sum(correct);
end