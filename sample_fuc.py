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
