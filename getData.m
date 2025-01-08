function [dataTable, data, x, y, xTrain, yTrain, xTest, yTest] = getData(fileName)
    dataTable = readtable(fileName);
    data = [];
    y = [];
    data(:,1) = dataTable(:,1).Variables;
    data(:,2) = dataTable(:,2).Variables;
    data(:,3) = (dataTable(:,3).Variables == "No")*0 + (dataTable(:,3).Variables == "Yes")*1;
    data(:,4) = dataTable(:,4).Variables;
    data(:,5) = dataTable(:,5).Variables;
    data(:,6) = dataTable(:,6).Variables;
 
    for i = [1:size(data,2)]
        data(:,i) = (data(:,i) - min(data(:,i))) / (max(data(:,i) - min(data(:,i))));
    end

    x = data(:,[1:end-1]);
    y = data(:, end);
    m = size(y);
    ratio = 0.8;
    cut = floor(ratio*m);
    xTrain = x([1:cut],:);
    yTrain = y([1:cut]);
    xTest = x([cut+1:m],:);
    yTest = y([cut+1:m]);
end
