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
JieBa = '../seperate/JIEBA.csv'
NlpIr = '../seperate/NLPIR.csv'
ThuLac = '../seperate/THULAC.csv'

ID = pd.read_csv(Id, encoding='GB18030')
AGE = pd.read_csv(Age, encoding='GB18030')
GENDER = pd.read_csv(Gender, encoding='GB18030')
EDUCATION = pd.read_csv(Education, encoding='GB18030')
JIEBA = pd.read_csv(JieBa, encoding='GB18030')
NLPIR = pd.read_csv(NlpIr, encoding='GB18030')
THULAC = pd.read_csv(ThuLac, encoding='GB18030')

df = pd.concat([ID, AGE, GENDER, EDUCATION, JIEBA, NLPIR, THULAC], axis=1)

print(df.shape)
re = df.duplicated().any()  # 重复值

mask = df.duplicated()
position = df[mask]
# print(repeat_position)  # 得到重复行是25175、47228、47229、56561、56562、58050、58051

# 按照ID列的值来删除重复行
df = df.drop_duplicates(subset=['ID'])

bk = df.isnull().any(axis=0)

bk_num = df.isnull().sum(axis=0)

bk_per = df.isnull().sum(axis=0) / df.shape[0]  # 各个变量缺失值比例


df_new = df.drop(labels=df.index[df['AGE'].isnull() |
                                 df['GENDER'].isnull() |
                                 df['QUERYLIST'].isnull() |
                                 df['JIEBA'].isnull() |
                                 df['NLPIR'].isnull() |
                                 df['THULAC'].isnull() |
                                 df['EDUCATION'].isnull()], axis=0)

bk_new = df_new.isnull().any(axis=0)


df_new['AGE'] = df_new['AGE'].astype(int)
df_new['GENDER'] = df_new['GENDER'].astype(int)
df_new['EDUCATION'] = df_new['EDUCATION'].astype(int)
print(df_new.dtypes)

# 删除 'Age' 列中值为 0 的数据
df_Ano = df_new[df_new['AGE'] != 0]
df_Gno = df_Ano[df_Ano['GENDER'] != 0]
df_Eno = df_Gno[df_Gno['EDUCATION'] != 0]


# 计算'JIEBA'列中每个元素的长度，并将结果写入新的列'JIEBA_Length'中
df_Eno['JIEBA_Length'] = df_Eno['JIEBA'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)
df_Eno['NLPIR_Length'] = df_Eno['NLPIR'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)
df_Eno['THULAC_Length'] = df_Eno['THULAC'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)

df_new.to_csv('../seperate/labels_all_0.csv', index=False, encoding='GB18030')
df_Eno.to_csv('../seperate/labels_all.csv', index=False, encoding='GB18030')
