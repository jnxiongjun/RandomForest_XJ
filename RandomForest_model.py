#!/usr/bin/python2.7
# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
import os
import random
import pickle
import sys
import time
start = time.clock()
#sys.path.append('lib_path')

'''
step1. 分别从正负样本中提取出10%的数据用于模型后期准确性验证，剩余90%用于模型的训练和测试
step2. 从上述90%中分别从正负样本中提取70%用于模型训练，另外的30%用于模型验证
step3. 重复进行step2 N次，形成 N的个模型，从N个模型中选出满足阈值的有效模型，最终模型通过集成有效模型加权得输出结果。
'''

#==============================================================================
# 数据集抽样
#==============================================================================

def validation(df, fraction, seed):
    '''

    :param df: 数据框
    :param fraction:  提取比例
    :param seed: 随机种子
    :return: 将一个样本集按照一定比例分成两部分   例如 利用从正例中或者负例中提取fraction比例的数据
    '''
    random.seed(seed)
    sc = int(df.shape[0] * fraction)

    #训练和测试样本
    ss = random.sample(range(df.shape[0]), sc)
    #验证样本
    rs = list(set(range(df.shape[0])) - set(ss))

    return df.iloc[rs,], df.iloc[ss,]
    # iloc 函数 df.iloc[ss,] 被抽中的


def split(positive, negative, seed, splits, factor=1):  # splits = split(pt, nt, i, 5)
    # """将训练集和测试机的目标变量分离出板""
    '''

    :param positive: 正例数据框
    :param negative: 负例数据框
    :param seed: 随机种子
    :param splits: 分割？？
    :param factor: 因子
    :return: 训练集和测试集
    '''
    #按照样本的数据条数，随机抽样20%作为测试集，剩下的作为训练集
    scount1 = int(negative.shape[0] / splits) + 1
    scount2 = int(positive.shape[0] / splits) + 1
    def _split(df, ssize, seed):
        random.seed(seed)
        total = df.shape[0]
        #样本抽样
        #vi抽样序列
        #ti抽样剩下的序列
        vi = random.sample(range(total), ssize)
        ti = list(set(range(total)) - set(vi))
        #返回一个元组，元组的值为两个数据框
        return df.iloc[ti,], df.iloc[vi,]
    #负样本抽样
    nt, nv = _split(negative, int(scount1 * factor), seed + 1)
    #正样本抽样
    pt, pv = _split(positive, scount2, seed + 2)

    #整合数据,在nv后加pv形成测试集vs,在nt后加pt形成训练集ts
    vs = nv.append(pv)
    ts = nt.append(pt)

    #返回ts的第一列之后作为X变量，第一列为Y变量 ，vs的第一列第二列,返回的为二维元组，存储的均为数据框
    return (ts.iloc[:, 1:], ts.iloc[:, :1]), (vs.iloc[:, 1:], vs.iloc[:, :1])
#==============================================================================
# 变量离散化——等宽分箱函数
#==============================================================================
def data_cut(dataSet):  
    m,n = np.shape(dataSet)    #获取数据集行列（样本数和特征数)  
    disMat = np.tile([0],np.shape(dataSet))  #初始化离散化数据集  
    for i in range(n):    #遍历特征列  
        x =[l[i] for l in dataSet] #获取第i+1特征向量  
        y = pd.cut(x,200,labels=range(200))   #调用cut函数，将特征离散化为10类，可根据自己需求更改离散化种类  
        for k in range(m):  #将离散化值传入离散化数据集  
            disMat[k][i] = y[k]      
    return disMat 

#==============================================================================
# 变量离散化——等频分箱函数
#==============================================================================
def data_qcut(dataSet):  
    m,n = np.shape(dataSet)    #获取数据集行列（样本数和特征数)  
    disMat = np.tile([0],np.shape(dataSet))  #初始化离散化数据集  
    for i in range(n):    #遍历特征列  
        x =[l[i] for l in dataSet] #获取第i+1特征向量  
        y = pd.qcut(x,int(m/2),labels=False)   #调用cut函数，将特征离散化为10类，可根据自己需求更改离散化种类  
        for k in range(m):  #将离散化值传入离散化数据集  
            disMat[k][i] = y[k]      
    return disMat 



#==============================================================================
# 导入数据
#==============================================================================
names=['bad1','contact_cnt','shanghai_cnt','yunnan_cnt','neimeng_cnt','beijing_cnt','jiling_cnt','sichuan_cnt','tianjing_cnt','ningxia_cnt','anhui_cnt','shangdong_cnt','sanxi_cnt','guangdong_cnt','guangxi_cnt','xinjiang_cnt','jiangsu_cnt','jiangxi_cnt','hebei_cnt','henan_cnt','zhejiang_cnt','hainan_cnt','hubei_cnt','hunan_cnt','gansu_cnt','fujian_cnt','xizang_cnt','guizhou_cnt','liaonin_cnt','chongqin_cnt','shanxi_cnt','qinhai_cnt','heilong_cnt','null_cnt','tuixiao_cnt','ys_tuixiao_cnt','saorao_cnt','zhapian_cnt','weifa_cnt','weizhi_cnt','same_cnt','GPS_type','gv_1','gv_2','gv_3','gv_4','gv_5','gv_6','gv_7','gv_8','gv_9','gv_10','gv_11','gv_12','gv_13','gv_14','gv_15','gv_16','gv_17','gv_18','gv_19','gv_20','gv_21','gv_22','gv_23','gv_24','gv_25','gv_26','gv_27','gv_28','gv_29','gv_30','gv_31','gv_32','gv_33','gv_34','gv_35','gv_36','gv_37','gv_38','gv_39','gv_40','gv_41','gv_42','gv_43','gv_44','gv_45','gv_46','gv_47','gv_48','gv_49','gv_50','corporation1','corporation2','corporation3','same_rate']
data_form = pd.read_csv('txl1208.dat',names=names,sep='\1')

#新增部分变量
#需要叠加的变量列表
columns1=['gv_1','gv_2','gv_3','gv_4','gv_5','gv_6','gv_7','gv_8','gv_9','gv_10','gv_11','gv_12','gv_13','gv_14','gv_15','gv_16','gv_17','gv_18','gv_19','gv_20','gv_21','gv_22','gv_23','gv_24','gv_25','gv_26','gv_27','gv_28','gv_29','gv_30','gv_31','gv_32','gv_33','gv_34','gv_35','gv_36','gv_37','gv_38','gv_39','gv_40','gv_41','gv_42','gv_43','gv_44','gv_45','gv_46','gv_47','gv_48','gv_49','gv_50']
columns2=['tuixiao_cnt','ys_tuixiao_cnt','saorao_cnt','zhapian_cnt','weifa_cnt']
columns3=['shandong_cnt','jiangsu_cnt','anhui_cnt','zhejiang_cnt','fujian_cnt','shanghai_cnt']
columns4=['guangdong_cnt','guangxi_cnt','hainan_cnt']
columns5=['hubei_cnt','hunan_cnt','henan_cnt','jiangxi_cnt']
columns6=['beijing_cnt','tianjing_cnt','hebei_cnt','sanxi_cnt','neimeng_cnt']
columns7=['ningxia_cnt','xinjiang_cnt','qinghai_cnt','shanxi_cnt','gansu_cnt']
columns8=['sichuan_cnt','yunnan_cnt','guizhou_cnt','xizang_cnt','chongqin_cnt']
columns9=['liaonin_cnt','jiling_cnt','heilong_cnt']
columns10=['gv_4','gv_6','gv_7','gv_8','gv_9','gv_10','gv_11','gv_13','gv_14','gv_15','gv_16','gv_17','gv_18','gv_19','gv_20','gv_21','gv_22','gv_23','gv_24','gv_25','gv_26','gv_27','gv_28','gv_35']

data_form['gv_sum'] = data_form.apply(lambda x: x[columns1].sum() , axis=1)
data_form['spyder_sum'] = data_form.apply(lambda x: x[columns2].sum() , axis=1)
data_form['huadong_zone'] = data_form.apply(lambda x: x[columns3].sum() , axis=1)
data_form['huanan_zone'] = data_form.apply(lambda x: x[columns4].sum() , axis=1)
data_form['huazhong_zone'] = data_form.apply(lambda x: x[columns5].sum() , axis=1)
data_form['huabei_zone'] = data_form.apply(lambda x: x[columns6].sum() , axis=1)
data_form['xibei_zone'] = data_form.apply(lambda x: x[columns7].sum() , axis=1)
data_form['xinan_zone'] = data_form.apply(lambda x: x[columns8].sum() , axis=1)
data_form['dongbei_zone'] = data_form.apply(lambda x: x[columns9].sum() , axis=1)
data_form['gv_51sum'] = data_form.apply(lambda x: x[columns10].sum() , axis=1)

#筛选部分变量
x_columns = [x for x in data_form.columns if x not in ['GPS_type','weizhi_cnt','gv_29','gv_34','gv_47','gv_40','gv_4','gv_20','gv_25','gv_16','gv_6','gv_14','gv_28','gv_22','gv_7','gv_8','gv_9','gv_10','gv_11','gv_13','gv_15','gv_17','gv_18','gv_19','gv_21','gv_23','gv_24','gv_26','gv_27','gv_31','gv_35']]
#x_columns = [x for x in data_form.columns if x in ['bad1','same_rate','same_cnt','corporation2','contact_cnt','corporation1','corporation3','null_cnt','guangdong_cnt','zhejiang_cnt','jiangsu_cnt','hubei_cnt','fujian_cnt','sichuan_cnt','hunan_cnt','guangxi_cnt','beijing_cnt','anhui_cnt','henan_cnt','shanghai_cnt','shangdong_cnt','hebei_cnt','shanxi_cnt','jiangxi_cnt','chongqin_cnt','gv_sum','yunnan_cnt','hainan_cnt','guizhou_cnt','sanxi_cnt','liaonin_cnt','heilong_cnt','gansu_cnt','neimeng_cnt','jiling_cnt','tianjing_cnt','xinjiang_cnt','spyder_sum','gv_32','gv_33','ningxia_cnt','tuixiao_cnt','gv_1','xizang_cnt','saorao_cnt','gv_48','gv_49','qinhai_cnt','gv_2','ys_tuixiao_cnt','gv_44','gv_41','gv_5','gv_3','weifa_cnt','zhapian_cnt','gv_30','gv_12','gv_42','gv_43','gv_45','gv_46','gv_40','gv_37','gv_39','huadong_zone','huanan_zone','huazhong_zone','huabei_zone','xibei_zone','xinan_zone','dongbei_zone']]

data = data_form[x_columns]

#删除极值
#data = data[data.contact_cnt<600]
#剔除通讯联系个数为1的数据
data = data[(data.contact_cnt>1) & (data.contact_cnt<600)]

#==============================================================================
# 特征列—变量离散化——等宽分箱/等频分箱
#==============================================================================
#==============================================================================
# feature_columns = [x for x in data.columns if x not in ['bad1']]
# feature_data = data[feature_columns]
# 
# #bad列重新赋值索引
# y=data['bad1'].values
# Y=pd.Series(y,index=range(feature_data.shape[0]))
# 
# 
# #特征列离散化
# feature_binning=data_cut(feature_data.values)
# feature_data_bin=pd.DataFrame(feature_binning,columns=feature_columns)
# 
# #特征列添加因变量bad，还原为原始数据
# #feature_data_bin['bad1']=Y
# data=feature_data_bin
# #插入变量至第一列
# data.insert(0,'bad1',Y)
#==============================================================================

#==============================================================================
# 样本抽样
#==============================================================================
positive = data[data.bad1 > 0]
negative = data[data.bad1 <= 0]

#正样本划分训练和验证,pt为正训练样本，pv为正验证样本
pt, pv = validation(positive, .2, 1)
#负样本划分训练和验证,nt为负训练样本，nv为负验证样本
nt, nv = validation(negative, .2, 2)


#==============================================================================
# 函数selection用于训练模型
#==============================================================================
def selection(repeat, percentile):

    models = []

    aucs = []
    for i in range(1, repeat + 1):
        if i % 50 == 0:
            print("Repeat for the %d-th time." % i)

        #训练集划分训练和测试
        splits = split(pt, nt, i, 5)
        #splits[1][0]为测试集的X变量，splits[1][1]为测试集的Y变量
        if splits[1][0].shape[0] == 0:
            continue

        rfc = RandomForestClassifier(n_estimators= 100,max_features=10)
#利用训练集建模,     ravel函数将多维数组进行铺平为一维数组   
        rfc.fit(splits[0][0], splits[0][1].values.ravel())
#预测测试集的数据
        prob = rfc.predict_proba(splits[1][0])[:, 1]  # splits[1][0]
#ROC曲线指标
        fpr, tpr, _ = roc_curve(splits[1][1], prob)
#计算AUC值
        roc_auc = auc(fpr, tpr)
#模型叠加,序列的形式
        models += [(rfc, roc_auc)]

        if roc_auc > .5:
            aucs += [roc_auc]

    t = np.percentile(aucs,percentile)

    #models以x的第一列为基础进行降序排列
    models = sorted(models, key=lambda x: x[1],
                    reverse=True)

    smodels = list((model, aval) for model, aval in models if aval > t)  # 选择模型
    # 存到文件
    tauc = sum(aval for model, aval in smodels)  # 计算roc_auc的和
    weights = list((model, aval / tauc) for model, aval in smodels)  # 生成 模型 和模型权重
    
    #各弱分类器赋予相应权重，叠加成强分类器
    #model_strong=[]
    #for j in range(0,len(weights)):
    #    model_strong=model_strong+weights[j][0]*weights[j][1]
    
    
    #二进制模式写入文件至New_Random.txt
    f = open('New_Random.txt', 'wb')
    #使用pickle模块将数据对象保存到文件
    pickle.dump(weights, f)
    f.close()
    #return model_strong
    return weights

#==============================================================================
# 弱分类器累加为强分类器，输出结果
#==============================================================================
def strong(weights,x):
        strong_prob = 0
        for m, w in weights:  # 取出模型 和模型的权重
            prob = m.predict_proba(x)[:, 1]  #  输入新纪录属性 特征
            strong_prob += prob * w
        return (strong_prob)

#==============================================================================
# 模型结果验证
#==============================================================================
result=selection(20,90)
#验证集验证数据预处理
data_test=pv.append(nv)

#用整体数据看随机森林的分类效果
#data_test=data
testx=data_test.iloc[:, 1:]
testy=data_test.iloc[:, :1]

#验证集预测数据,result[0]表示第一个模型+其AUC值，result[0][0]表示取出第一个模型
#强分类器的输出
strong_output=strong(result,testx)

predicty=np.zeros(strong_output.shape[0])

for i in range(strong_output.shape[0]) :
    if strong_output[i]>0.65:
        predicty[i]=1
    else:
        predicty[i]=0

predicty=list(predicty)
               
#取AUC最大值的模型进行预测                
#predicty=list(result[0][0].predict(testx))
#predicty=list(result.predict(testx))

#==============================================================================
# 验证样本ROC与AUC
#==============================================================================
prob_test = result[0][0].predict_proba(testx)[:, 1]
#方法一、先算ROC,再算AUC
# 验证样本ROC曲线指标
#fpr_test, tpr_test, _ = roc_curve(testy, prob_test)
#计算验证样本AUC值
#roc_auc1 = auc(fpr_test, tpr_test)

#方法二、直接算AUC分数
roc_auc2=roc_auc_score(testy,prob_test)


#将testy数据框转化为数组进行处理
testy_array=testy.bad1.values
testy_list=testy_array.tolist()

#print('模型测试时的AUC分数为%.3f'%result[0][1])
print('模型验证时的AUC分数为%.3f'%roc_auc2)
print('验证样本的正样本占比为%.2f%%'%(testy_list.count(1)*100/len(testy_list)))
print('预测结果中含正样本数为',predicty.count(1))

a=[[0,0],[0,0]]

for i in range(0,len(predicty)):
    if predicty[i]==0 and testy_array[i]==0:
        a[0][0]=a[0][0]+1
    elif predicty[i]==1 and testy_array[i]==0:
        a[1][0]=a[1][0]+1
    elif predicty[i]==0 and testy_array[i]==1:
        a[0][1]=a[0][1]+1
    else:
        a[1][1]=a[1][1]+1

print('验证样本的结果为',a)
print('模型在验证样本的【准确率/精确度】为%.2f%%'%(a[1][1]*100/(a[1][0]+a[1][1])))
print('             【打扰率/虚警率】为%.2f%%'%(a[1][0]*100/(a[1][0]+a[0][0])))
print('             【覆盖率/召回率】为%.2f%%'%(a[1][1]*100/(a[1][1]+a[0][1])))

#计算运行时间
end = time.clock()

print ('running time :%.3f'%(end-start),'s')
