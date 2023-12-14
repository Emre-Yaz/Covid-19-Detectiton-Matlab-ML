%% Importing data
clear
close all

d_set = readtable("C:\Users\eyaz9\Desktop\bitirme\dataset\covid_dataset.csv", ...
    'VariableNamingRule', 'preserve');
label_set = d_set(:,21);
d_set = d_set(:,1:20);

%% K Fold
f1test = d_set(1:1087,:);
f2test = d_set(1088:2174,:);
f3test = d_set(2175:3261,:);
f4test = d_set(3262:4348,:);
f5test = d_set(4349:5434,:);

l1test = label_set(1:1087,:);
l2test = label_set(1088:2174,:);
l3test = label_set(2175:3261,:);
l4test = label_set(3262:4348,:);
l5test = label_set(4349:5434,:);

f1train = d_set(1088:5434,:);
f2train = [f1test;f3test;f4test;f5test];
f3train = [f1test;f2test;f4test;f5test];
f4train = [f1test;f2test;f3test;f5test];
f5train = [f1test;f2test;f3test;f4test];

l1train = label_set(1088:5434,:);
l2train = [l1test;l3test;l4test;l5test];
l3train = [l1test;l2test;l4test;l5test];
l4train = [l1test;l2test;l3test;l5test];
l5train = [l1test;l2test;l3test;l4test];

%% Training The SVM Model

%f1test
f1svmmodel = fitcsvm(f1train,l1train,'KernelScale',6);
f1predictions = predict(f1svmmodel,f1test);
f1iscorrect = (f1predictions == l1test.COVID19);
f1accuracy = sum(f1iscorrect)/numel(f1predictions);

%f2test
f2svmmodel = fitcsvm(f2train,l2train,'KernelScale',6);
f2predictions = predict(f2svmmodel,f2test);
f2iscorrect = (f2predictions == l2test.COVID19);
f2accuracy = sum(f2iscorrect)/numel(f2predictions);

%f3test
f3svmmodel = fitcsvm(f3train,l3train,'KernelScale',6);
f3predictions = predict(f3svmmodel,f3test);
f3iscorrect = (f3predictions == l3test.COVID19);
f3accuracy = sum(f3iscorrect)/numel(f3predictions);

%f4test
f4svmmodel = fitcsvm(f4train,l4train,'KernelScale',6);
f4predictions = predict(f4svmmodel,f4test);
f4iscorrect = (f4predictions == l4test.COVID19);
f4accuracy = sum(f4iscorrect)/numel(f4predictions);

%f5 test
f5svmmodel = fitcsvm(f5train,l5train,'KernelScale',6);
f5predictions = predict(f5svmmodel,f5test);
f5iscorrect = (f5predictions == l5test.COVID19);
f5accuracy = sum(f5iscorrect)/numel(f5predictions);

totalAcc = (f1accuracy+f2accuracy+f3accuracy+f4accuracy+f5accuracy)/5

%% Performance Metrics
%kernelscale1acc = 0.8617
%kernelscale2acc = 0.8545
%kernelscale3acc = 0.8643
%kernelscale4acc = 0.8687
%kernelscale5acc = 0.8770
%kernelscale6acc = 0.8797 !!!!!!!!!!!!
%kernelscale7acc = 0.8755
%kernelscale20acc = 0.8543

cc1 = confusionchart(l1test.COVID19,f1predictions);
cc1.ColumnSummary = 'column-normalized';
cc1.RowSummary = 'row-normalized';
cc1.Title = 'Covid-19 Confusion Chart 1';

tp1 = cc1.NormalizedValues(1,1);
fp1 = cc1.NormalizedValues(1,2);
fn1 = cc1.NormalizedValues(2,1);
tn1 = cc1.NormalizedValues(2,2);

cc2 = confusionchart(l2test.COVID19,f2predictions);
cc2.ColumnSummary = 'column-normalized';
cc2.RowSummary = 'row-normalized';
cc2.Title = 'Covid-19 Confusion Chart 2';

tp2 = cc2.NormalizedValues(1,1);
fp2 = cc2.NormalizedValues(1,2);
fn2 = cc2.NormalizedValues(2,1);
tn2 = cc2.NormalizedValues(2,2);

cc3 = confusionchart(l3test.COVID19,f3predictions);
cc3.ColumnSummary = 'column-normalized';
cc3.RowSummary = 'row-normalized';
cc3.Title = 'Covid-19 Confusion Chart 3';

tp3 = cc3.NormalizedValues(1,1);
fp3 = cc3.NormalizedValues(1,2);
fn3 = cc3.NormalizedValues(2,1);
tn3 = cc3.NormalizedValues(2,2);

cc4 = confusionchart(l4test.COVID19,f4predictions);
cc4.ColumnSummary = 'column-normalized';
cc4.RowSummary = 'row-normalized';
cc4.Title = 'Covid-19 Confusion Chart 4';

tp4 = cc4.NormalizedValues(1,1);
fp4 = cc4.NormalizedValues(1,2);
fn4 = cc4.NormalizedValues(2,1);
tn4 = cc4.NormalizedValues(2,2);

cc5 = confusionchart(l5test.COVID19,f5predictions);
cc5.ColumnSummary = 'column-normalized';
cc5.RowSummary = 'row-normalized';
cc5.Title = 'Covid-19 Confusion Chart 5';

tp5 = cc5.NormalizedValues(1,1);
fp5 = cc5.NormalizedValues(1,2);
fn5 = cc5.NormalizedValues(2,1);
tn5 = cc5.NormalizedValues(2,2);

tp = tp1 + tp2 + tp3 + tp4 + tp5
fn = fn1 + fn2 + fn3 + fn4 + fn5
fp = fp1 + fp2 + fp3 + fp4 + fp5
tn = tn1 + tn2 + tn3 + tn4 + tn5

tp + fn + fp + tn

precision = tp/(tp+fp)

recall = tp/(tp+fn)

specificity = tn/(tn+fp)

f1score = (2*precision*recall) / (precision + recall)

%% ROC
[labels1, scores1] = resubPredict(f1svmmodel);
[X1,Y1,T1,AUC1] = perfcurve(f1svmmodel.Y,scores1(:,1),"0");
plot(X1,Y1)
title('ROC 1')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
%hold on

[labels2, scores2] = resubPredict(f2svmmodel);
[X2,Y2,T2,AUC2] = perfcurve(f2svmmodel.Y,scores2(:,1),"0");
plot(X2,Y2)
title('ROC 2')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
%hold on

[labels3, scores3] = resubPredict(f3svmmodel);
[X3,Y3,T3,AUC3] = perfcurve(f3svmmodel.Y,scores3(:,1),"0");
plot(X3,Y3)
title('ROC 3')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
%hold on

[labels4, scores4] = resubPredict(f4svmmodel);
[X4,Y4,T4,AUC4] = perfcurve(f4svmmodel.Y,scores4(:,1),"0");
plot(X4,Y4)
title('ROC 4')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
%hold on

[labels5, scores5] = resubPredict(f5svmmodel);
[X5,Y5,T5,AUC5] = perfcurve(f5svmmodel.Y,scores5(:,1),"0");
plot(X5,Y5)
title('ROC 5')
xlabel('False Positive Rate')
ylabel('True Positive Rate')

%title('ROC Curves')  %%% To see all curves together.(Do not forget to uncomment 'hold on'.)
%legend('Roc1','Roc2','Roc3','Roc4','Roc5')
%hold off

AUCavg = (AUC1 + AUC2 + AUC3 + AUC4 + AUC5) / 5
