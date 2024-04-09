"""
将不同的label文件合并成一个表格;
后续用该csv文件可以做描述性统计;
也可以做数据视觉化分析,得到用户群体的初步画像;
"""
'''
检查数据中是否还有空白值，重复值；进一步预处理
'''

import pandas as pd

Id = '../Train/ID.csv'
Age = '../Train/AGE.csv'
Gender = '../Train/GENDER.csv'
Education = '../Train/EDUCATION.csv'
# QueryList = '../Train/Process_Query.csv'
JieBa = '../Train/JIEBA.csv'
NlpIr = '../Train/NLPIR.csv'
ThuLac = '../Train/THULAC.csv'

ID = pd.read_csv(Id, encoding='GB18030')
AGE = pd.read_csv(Age, encoding='GB18030')
GENDER = pd.read_csv(Gender, encoding='GB18030')
EDUCATION = pd.read_csv(Education, encoding='GB18030')
JIEBA = pd.read_csv(JieBa, encoding='GB18030')
NLPIR = pd.read_csv(NlpIr, encoding='GB18030')
THULAC = pd.read_csv(ThuLac, encoding='GB18030')

df = pd.concat([ID, AGE, GENDER, EDUCATION, JIEBA, NLPIR, THULAC], axis=1)

data = pd.concat([ID, AGE, GENDER, EDUCATION], axis=1)

print(data.shape)  # 共99859行,9列
print(df.shape)
repeat = data.duplicated().any()  # 重复值
re = df.duplicated().any()  # 重复值
print(repeat)  # 结果为False
# print('原来数据表')
# print(data.shape)

repeat_mask = data.duplicated()
mask = df.duplicated()
repeat_position = data[repeat_mask]
position = df[mask]
# print(repeat_position)  # 得到重复行是25175、47228、47229、56561、56562、58050、58051

# 按照ID列的值来删除重复行
data = data.drop_duplicates(subset=['ID'])
df = df.drop_duplicates(subset=['ID'])
# print('第一次修改后')
# print(data.shape)

blank = data.isnull().any(axis=0)  # 看data中是否有空白的
bk = df.isnull().any(axis=0)
print(blank)  # 缺失值

blank_number = data.isnull().sum(axis=0)  # 各个变量缺失值数量
bk_num = df.isnull().sum(axis=0)
print(blank_number)  # 空白值

blank_percent = data.isnull().sum(axis=0) / data.shape[0]  # 各个变量缺失值比例
bk_per = df.isnull().sum(axis=0) / data.shape[0]  # 各个变量缺失值比例
print(blank_percent)  # 缺失值占比

# 删除年龄/性别/受教育程度中的缺失值
data_new = data.drop(labels=data.index[data['AGE'].isnull() |
                                       data['GENDER'].isnull() |
                                       data['EDUCATION'].isnull()], axis=0)

df_new = df.drop(labels=df.index[df['AGE'].isnull() |
                                 df['GENDER'].isnull() |
                                 df['QUERYLIST'].isnull() |
                                 df['JIEBA'].isnull() |
                                 df['NLPIR'].isnull() |
                                 df['THULAC'].isnull() |
                                 df['EDUCATION'].isnull()], axis=0)
# print(data_new.shape)

blank_new = data_new.isnull().any(axis=0)  # 看data中是否有空白的
bk_new = df_new.isnull().any(axis=0)
print(blank_new)  # 返回结果都为False,证明新的数据表已处理完成

data_new['AGE'] = data_new['AGE'].astype(int)
data_new['GENDER'] = data_new['GENDER'].astype(int)
data_new['EDUCATION'] = data_new['EDUCATION'].astype(int)
df_new['AGE'] = df_new['AGE'].astype(int)
df_new['GENDER'] = df_new['GENDER'].astype(int)
df_new['EDUCATION'] = df_new['EDUCATION'].astype(int)
print(data_new.dtypes)
print(df_new.dtypes)

# 删除 'Age' 列中值为 0 的数据
data_Ano = data_new[data_new['AGE'] != 0]
data_Gno = data_Ano[data_Ano['GENDER'] != 0]
data_Eno = data_Gno[data_Gno['EDUCATION'] != 0]

df_Ano = df_new[df_new['AGE'] != 0]
df_Gno = df_Ano[df_Ano['GENDER'] != 0]
df_Eno = df_Gno[df_Gno['EDUCATION'] != 0]

print(data_Eno.describe())  # 对标签进行描述性统计

data_new.to_csv('../label/labels_analyse_0.csv', index=False, encoding='GB18030')
data_Eno.to_csv('../label/labels_analyse.csv', index=False, encoding='GB18030')
df_new.to_csv('../label/labels_all_0.csv', index=False, encoding='GB18030')
df_Eno.to_csv('../label/labels_all.csv', index=False, encoding='GB18030')
