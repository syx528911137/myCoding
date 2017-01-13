#-*-coding:UTF-8-*-
import misvm
import random
import numpy



def readData(filePath):
    f = open(filePath,'r')
    lineContent = f.readline()
    data = []
    count =0
    while len(lineContent) != 0:
        count = count + 1
        print str(count)
        example = []
        lineContent = lineContent.strip()
        tmp_data = lineContent.split(";")
        tmp_ins = tmp_data[2].split("/")
        bag = []
        label = int(tmp_data[1])
        for i in range(0,len(tmp_ins)):
            instance = []
            tmp = tmp_ins[i].split(" ")
            for j in range(0,len(tmp)):
                instance.append(float(tmp[j]))

            bag.append(instance)
        example.append(bag)
        example.append(label)
        data.append(example)
        lineContent = f.readline()
    return data



def getPrecision(pred_y,true_y):
    all = 0.0
    correct = 0.0
    for i in range(0,len(true_y)):
        all = all + 1.0
        if float(pred_y[i]) == float(true_y[i]):
            correct = correct + 1.0
    p = correct / all
    return  p


def getRecall(pred_y,true_y):
    all = 0.0
    correct = 0.0
    for i in range(0,len(true_y)):
        if float(true_y[i]) == 1.0:
            all = all + 1.0
            if float(true_y[i]) == float(pred_y[i]):
                correct = correct + 1.0
    r = correct / all
    return  r




if __name__ == '__main__':
    filePath = 'dataSet20news_MI.txt'
    wfilePath = 'resultOfLDA20news_miSVM.txt'
    wfile = open(wfilePath,'w')
    data = readData(filePath)
    scale = 5
    allLabelP = []
    allLabelR = []
    allLabelF = []
    for i in range(1,21): # for each label
        usedData = []
        positiveExamples = []
        negativeExamples = []
        print "label:" + str(i)
        print "process data..."
        for j in range(0,len(data)):
            if data[j][1] == i:
                tmpdata = data[j]
                tmpdata[1] = 1
                positiveExamples.append(tmpdata)
            else:
                tmpdata = data[j]
                tmpdata[1] = -1
                negativeExamples.append(tmpdata)
        print "generate useddata...."
        for exam in positiveExamples:
            usedData.append(exam)
        sample = random.sample(negativeExamples,len(positiveExamples) * scale)

        for exam in sample:
            usedData.append(exam)
        print "random sort..."
        random.shuffle(usedData)
        fold_precision = 0.0
        fold_recall = 0.0
        fold_fscore = 0.0

        for fold in range(0,10): # 10 fold validate

            trainData = []
            trainLabel = []
            testData = []
            testLabel = []
            chunkSize = int(len(usedData) / 10)
            print "generate training and testing data..."
            for k in range(0,len(usedData)):
                if k >= fold*chunkSize and k < (fold + 1) * chunkSize:
                    testData.append( usedData[k][0])
                    testLabel.append( usedData[k][1])
                else:
                    trainData.append( usedData[k][0])
                    trainLabel.append(usedData[k][1])

            print "training model..."
            clf = misvm.MISVM(kernel='linear',C= 1.0, max_iters=50)
            clf.fit(trainData,trainLabel)
            print "predict..."
            print "testdata size:" + str(len(testData))
            pred_y = []
            for bag in testData:
                pred_y.append(numpy.sign(clf.predict(bag)))
            #pred_y = clf.predict(testData)
            p = getPrecision(pred_y,testLabel)
            r = getRecall(pred_y,testLabel)
            f1 = 2 * p * r / (p + r)
            fold_precision = fold_precision + p
            fold_recall = fold_recall + r
            fold_fscore = fold_fscore + f1
            print "label:" + str(i) + ", fold:" + str(fold) + ", precision:" + str(p) + ", recall:" + str(r) + ", f1-score:" + str(f1)
        fold_precision = fold_precision / 10.0
        fold_recall = fold_recall / 10.0
        fold_fscore = fold_fscore / 10.0
        allLabelP.append(fold_precision)
        allLabelR.append(fold_recall)
        allLabelF.append(fold_fscore)
        wContent = "label:" + str(i)  + ", precision:" + str(fold_precision) + ", recall:" + str(fold_recall) + ", f1-score:" + str(fold_fscore)
        wfile.write(wContent)
        wfile.write('\n')
        wfile.flush()
    _p = 0.0
    _r = 0.0
    _f = 0.0
    for m in range(0,len(allLabelP)):
        _p = _p + allLabelP[m]
        _r = _r + allLabelR[m]
        _f = _f + allLabelF[m]
    _p = _p / len(allLabelP)
    _r = _r / len(allLabelP)
    _f = _f /len(allLabelF)
    w = "precision:" + str(_p) + ", recall:" + str(_r) + ", f1-score:" + str(_f)
    wfile.write(w)
    wfile.close()



