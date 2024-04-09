import pandas as pd
import jieba

# ----------------------load data--------------------------------
trainname = '../Train/Process_Query.csv'
stopword_path = '../stopwords/cn_stopwords.txt'
jieba_path = '../seperate/JIEBA.csv'
no_stop_path = '../no_stop/JIEBA.csv'
# 文本清洗
'''
打开停用词txt，并读取
'''

# 读取 CSV 文件并处理文本数据
data = pd.read_csv(trainname, encoding='GB18030')
# 停用词列表示例
with open(stopword_path, encoding='GB18030') as f:  # 可根据需要打开停用词库，然后加上不想显示的词语
    con = f.readlines()
    stop_words = set()  # 集合可以去重
    for i in con:
        i = i.replace("\n", "")  # 去掉读取每一行数据的\n
        stop_words.add(i)


def sep(text):
    # 分词
    words = jieba.cut(text)
    # 正则化
    regulation = ['[', ']', '.', '?', '!', "'", '"']
    words = [
        word for word in words
        if word not in regulation
        if word not in stop_words
    ]
    # words = " ".join(words)
    # 同义词替换
    return ' '.join(words)



def no_stop(text):
    # 分词
    words = jieba.cut(text)
    # 正则化
    regulation = ['[', ']', '.', '?', '!', "'", '"']
    words = [
        word for word in words
        if word not in regulation
        if word not in stop_words
    ]
    # words = " ".join(words)
    # 同义词替换
    return ' '.join(words)


# data['JIEBA'] = data['QUERYLIST'].apply(lambda x: sep(str(x)))
data['JIEBA'] = data['QUERYLIST'].apply(lambda x: no_stop(str(x)))
# 将处理后的数据保存到新的 CSV 文件中
# data.to_csv(jieba_path, index=False, encoding='GB18030')
data.to_csv(no_stop_path, index=False, encoding='GB18030')
