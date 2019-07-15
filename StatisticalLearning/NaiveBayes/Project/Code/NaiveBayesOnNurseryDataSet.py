#训练数据和测试数据以及数据的总数
trainingNum = 12000
testingNum = 960
totalNum = 12960
featureNum = 8

#传入数据文件名，进行处理，返回训练集和测试集
def fileToMatrix(filename):
   fr = open(filename, 'r', encoding='utf-8')
   #逐行读取数据文件
   arrayOfLines = fr.readlines()
   #初始化测试集和训练集
   dataSet = []
   testSet = []
   #逐行处理数据，形成训练集
   for i in range(trainingNum):
      arrayOfLines[i] = arrayOfLines[i].strip()
      listFromLine = arrayOfLines[i].split(',')
      dataSet.append(listFromLine)
   #逐行处理数据，形成测试集
   for i in range(testingNum):
      arrayOfLines[i+testingNum] = arrayOfLines[i+testingNum].strip()
      listFromLine = arrayOfLines[i+testingNum].split(',')
      testSet.append(listFromLine)
   #返回测试集和训练集
   return dataSet, testSet

#初始化分类器，即特征字典列表
def creatDictListOfFeature():
   dictListOfFeature = []
   #zeros为包含5个0的列表
   zeros = [0 for _ in range(5)]
   #每个特征字典的键为特征可能的取值，对应的键值初始化为一个包含5个0的列表
   dictListOfFeature.append({"usual":zeros[:], "pretentious":zeros[:], "great_pret":zeros[:]})
   dictListOfFeature.append({"proper":zeros[:], "less_proper":zeros[:], "improper":zeros[:],"critical":zeros[:], "very_crit":zeros[:]})
   dictListOfFeature.append({"complete":zeros[:], "completed":zeros[:], "incomplete":zeros[:],"foster":zeros[:]})
   dictListOfFeature.append({"1":zeros[:], "2":zeros[:], "3":zeros[:],"more":zeros[:]})
   dictListOfFeature.append({"convenient":zeros[:], "less_conv":zeros[:], "critical":zeros[:]})
   dictListOfFeature.append({"convenient":zeros[:], "inconv":zeros[:]})
   dictListOfFeature.append({"nonprob":zeros[:], "slightly_prob":zeros[:], "problematic":zeros[:]})
   dictListOfFeature.append({"recommended":zeros[:], "priority":zeros[:], "not_recom":zeros[:]})
   #返回分类器
   return dictListOfFeature

#传入训练集、分类器训练分类器
def trainNB(dataSet, dictListOfFeature, labelsSum, labels):
   #遍历每组数据，统计每个特征取值所对应的标签计数
   for i in range(trainingNum):
      #记录这组数据的标签索引
      labelIndex = labels.index(dataSet[i][-1])
      #记录此标签出现的次数
      labelsSum[labelIndex] += 1
      #遍历每个特征
      for j in range(featureNum):
         #统计此特征的取值所对应的标签计数
         featureValue = dataSet[i][j]
         dictListOfFeature[j][featureValue][labelIndex] += 1
   #将标签计数经过计算转化为概率
   for i in range(featureNum):
      for k in dictListOfFeature[i].keys():
         for j in range(5):
            #对于每个特征取值，将5个标签取值分别与5个标签总数相除，得到后验概率，引入了拉普拉斯平滑，分子加一，分母加此属性可能的取值数
            dictListOfFeature[i][k][j] = (dictListOfFeature[i][k][j]+1) / float(labelsNum[j]+len(dictListOfFeature[i]))
   return

#传入分类器和类先验概率labelsProb以及测试数据，进行分类并返回分类结果
def classify(dictListOfFeature, labelsProb, testVec):
   #用probList来统计属于每个标签取值的概率，初始化为0
   probList = [0 for _ in range(5)]
   #对于每个标签取值进行遍历，计算概率
   for i in range(5):
      #先取类先验概率，表示不考虑特征的情况下，取此种标签取值的概率
      #也就是在训练集中此种标签取值出现的概率
      probList[i] = labelsProb[i]
      #遍历每个特征，得到此组数据取这个标签取值的概率
      for j in range(featureNum):
         #分别乘以特征的条件概率，即在属于此标签取值的前提下这个特征取这个取值的概率
         probList[i] *= dictListOfFeature[j][testVec[j]][i]
   #求出概率最高的标签取值，作为分类结果
   maxProb = max(probList)
   #返回分类结果的索引
   return probList.index(maxProb)

#主函数
if __name__ == "__main__":
   #初始化训练集、测试集以及分类器
   filename = "../DataSets/nursery.data"
   dataSet, testSet = fileToMatrix(filename)
   dictListOfFeature = creatDictListOfFeature()
   #初始化标签取值计数列表和标签先验概率列表以及标签取值列表
   labelsNum = [0 for _ in range(5)]
   labelsProb = [0 for _ in range(5)]
   labels = ["not_recom", "recommend", "very_recom", "priority", "spec_prior"]
   #训练贝叶斯分类器
   trainNB(dataSet, dictListOfFeature, labelsNum, labels)
   #计算每个标签取值的先验概率，及在训练集中所出现的概率
   #引入了拉普拉斯平滑，分子加一，分母加标签取值数
   for i in range(5):
      labelsProb[i] = (labelsNum[i]+1) / float(trainingNum+5)
   #遍历测试集，分类每组数据并统计错误结果
   errorNum = 0
   for i in range(testingNum):
      #对每组数据进行分类
      classifyResult = classify(dictListOfFeature, labelsProb, testSet[i])
      #打印分类结果和实际结果
      print("预测结果为%s,实际结果为%s" % (labels[classifyResult], testSet[i][-1]))
      #若不一致则记录下来
      if(classifyResult != labels.index(testSet[i][-1])):
         errorNum += 1
   #输出测试结果
   print("一共%d组数据，其中%d组数据用于训练，%d组数据用于测试" % (totalNum, trainingNum, testingNum))
   print("一共%d组测试数据，判断错误%d组数据，错误率为%.3f%%" % (testingNum, errorNum, errorNum / testingNum * 100))