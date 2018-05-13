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
    