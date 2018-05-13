# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 11:35:55 2017

@author: xiongjun
"""
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
from scipy  import stats
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

    #验证样本
    ss = random.sample(range(df.shape[0]), sc)
    #训练和测试样本
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

        rfc = RandomForestClassifier(n_estimators= 100,max_depth=50)
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

# 弱分类器累加为强分类器，输出结果
#==============================================================================
def strong(weights,x):
        strong_prob = 0
        for m, w in weights:  # 取出模型 和模型的权重
            prob = m.predict_proba(x)[:, 1]  #  输入新纪录属性 特征
            strong_prob += prob * w
        return (strong_prob)


#==============================================================================
# 变量处理
#==============================================================================
# 变量离散化——等宽分箱函数
#==============================================================================
def data_cut(dataSet):  
    m,n = np.shape(dataSet)    #获取数据集行列（样本数和特征数)  
    disMat = np.tile([0],np.shape(dataSet))  #初始化离散化数据集  
    for i in range(n):    #遍历特征列  
        x =[l[i] for l in dataSet] #获取第i+1特征向量  
        y = pd.cut(x,20,labels=range(20))   #调用cut函数，将特征离散化为10类，可根据自己需求更改离散化种类  
        for k in range(m):  #将离散化值传入离散化数据集  
            disMat[k][i] = y[k]      
    return disMat 

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

#数据清洗，删除空值、单一值过多的变量
def variable_cleaning(df, keep_list, null_cutoff = 0.7, value_cutoff = 0.8, level_cutoff = 40):
    '''
    Function:
    =========
    Delete varaibles:
    1. with too many missings;
    2. with dominant values;
    3. with too many levels.

    Parameters:
    ===========
    df          : DataFrame, original data set
    keep_list   :      list, necessary variables to keep (like target and primary keys)
    null_cutoff :     float, cutoff for variables with missing values (default 0.7)
    value_cutoff:     float, cutoff for variables with dominant values(default 0.8)
    level_cutoff:   integer, cutoff for variables with multiple levels(default 40)

    Output:
    =======
    DataFrame: Cleaned data

    '''
    n_sample = df.shape[0]

    # Step 1: drop variables with too many missing values
    var_list_s1 = [x for x in df.columns if x not in keep_list]
    var_cal_s1 = df[var_list_s1]

    var_missing = var_cal_s1.apply(lambda x: x.isnull().sum(), axis = 0)/n_sample
    missing_list = list(var_missing[var_missing >= null_cutoff].index)

    df = df.drop(missing_list, axis = 1)

    # Step 2: drop variables with dominant values
    var_list_s2 = [x for x in df.columns if x not in keep_list]
    var_cal_s2 = df[var_list_s2]

    var_dom = var_cal_s2.apply(lambda x: (x.value_counts()/n_sample).max(), axis = 0)
    dom_list = list(var_dom[var_dom >= value_cutoff].index)

    df = df.drop(dom_list, axis = 1)

    # Step 3: deal with vairables with too many levels
    object_list = list(df.dtypes[df.dtypes == 'object'].index)
    var_list_s3 = [x for x in object_list if x not in keep_list]

    var_cal_s3 = df[var_list_s3]

    var_level = var_cal_s3.apply(lambda x: x.value_counts().count(), axis = 0)
    level_list = list(var_level[var_level >= level_cutoff].index)

    df = df.drop(level_list, axis = 1)

    return df


#==============================================================================
# 模型验证
#==============================================================================
#==============================================================================
#KS值计算
def KS_calculation(y, y_pred, bin_num=100):
    '''
    Calculate KS
    '''
    df = pd.DataFrame()
    df['y'] = y
    df['y_pred'] = y_pred
    n_sample = df.shape[0]
    y_cnt = df['y'].sum()
    bucket_sample = np.ceil(n_sample/bin_num)

    df = df.sort_values('y_pred', ascending = False)
    df['group'] = [np.ceil(x/bucket_sample) for x in range(1, n_sample+1)]

    grouped = df.groupby('group')['y'].agg({'Totalcnt': 'count',
                                            'Y_rate': np.mean,
                                            'Y_pct': lambda x: np.sum(x / y_cnt),
                                            'n_Y_pct': lambda x: np.sum((1-x) / (n_sample-y_cnt))})
    grouped['Cum_Y_pct'] = grouped['Y_pct'].cumsum()
    grouped['Cum_nY_pct'] = grouped['n_Y_pct'].cumsum()
    grouped['KS'] = (grouped['Cum_Y_pct'] - grouped['Cum_nY_pct']).map(lambda x: abs(x))

    KS = grouped['KS'].max()

    return KS
 
#混淆矩阵
def confusion_matrix(true,predicty):
    a=[[0,0],[0,0]]
    for i in range(0,predicty.shape[0]):
        if predicty[i]==0 and true[i]==0:
            a[0][0]=a[0][0]+1
        elif predicty[i]==1 and true[i]==0:
            a[1][0]=a[1][0]+1
        elif predicty[i]==0 and true[i]==1:
            a[0][1]=a[0][1]+1
        else:
            a[1][1]=a[1][1]+1
    FN=a[0][0]
    FP=a[1][0]
    TN=a[0][1]
    TP=a[1][1]
    return  (FN,TN,FP,TP)
