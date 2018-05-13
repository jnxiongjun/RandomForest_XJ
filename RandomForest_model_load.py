#
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:36:42 2017

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
start = time.clock()
#==============================================================================
#导入验证数据
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
#data = data[(data.contact_cnt>1) & (data.contact_cnt<600)]


#==============================================================================
#调用训练好的模型 
#==============================================================================
#使用pickle模块从文件中重构python对象
model_file = open('D:/python/RForest_modle/Good_modle/New_Random4-zq55%-fg2.45%.txt', 'rb')
result = pickle.load(model_file)
print(result)
model_file.close()
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
# 模型输出结果
#==============================================================================
data_test=data
testx=data_test.iloc[:, 1:]
testy=data_test.iloc[:, :1]
#强分类器的预测结果输出
strong_output=strong(result,testx)

predicty=np.zeros(strong_output.shape[0])

for i in range(strong_output.shape[0]) :
    if strong_output[i]>0.65:
        predicty[i]=1
    else:
        predicty[i]=0

predicty=list(predicty)

#==============================================================================
# 模型效果评估
#==============================================================================
testy_array=testy.bad1.values
testy_list=testy_array.tolist()

#print('模型测试时的AUC分数为%.3f'%result[0][1])
#print('模型验证时的AUC分数为%.3f'%roc_auc2)
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


