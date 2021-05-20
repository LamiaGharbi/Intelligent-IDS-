#%% Load data
import pandas as pd
X_test= pd.read_csv(r"newdataset\test_attack.csv",sep=';')
Y_test= pd.read_csv(r"newdataset\test_label.csv")
print(X_test.shape)
print(Y_test.shape)
#%% Test SOM:
num_test_data = (X_test.shape[0]);
print(num_test_data)
correc
t = 0; #%init
for i in enumerate(X_test):
    input = X_test[:,i];
    test_label = char(test_classlabel(i));
    #% Find winner neuron
    min_dist = inf;
    winner_r = -1; winner_c = -1;
    for r =1 : size(w,1)
        for c = 1 : size(w,2)
            dist = norm(input - shiftdim(w(r,c,:)),2);
            if (dist < min_dist)
                min_dist = dist;
                winner_r = r; winner_c = c;
            end
        end
    end #% END winner neuron found.
    output_label = char(neuron_labels(winner_r,winner_c));
    #% compare output label with test label.
    fprintf('output:%s ',output_label);
    fprintf('testlabel:%s ',test_label);
    if(output_label == test_label)
        fprintf ('CORRECT!\n');
        correct = correct + 1;
    else
        fprintf('WRONG!\n');
    end
end
accuracy = correct / num_test_data;
fprintf('Out of %d test data, %d were correctly classified.\n', num_test_data, correct);
fprintf('Accuracy: %f\n', accuracy);