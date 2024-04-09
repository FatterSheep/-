import pandas as pd
import pynlpir

trainname = '../Train/Process_Query.csv'
stopword_path = '../stopwords/cn_stopwords.txt'
nlpir_path = '../Train/NLPIR.csv'
NlpIr = '../no_stop/NLPIR.csv'

data = pd.read_csv(trainname, encoding='GB18030')
# 停用词列表示例
with open(stopword_path, encoding='GB18030') as f:  # 可根据需要打开停用词库，然后加上不想显示的词语
    con = f.readlines()
    stop_words = set()  # 集合可以去重
    for i in con:
        i = i.replace("\n", "")  # 去掉读取每一行数据的\n
        stop_words.add(i)

# 初始化 NLPIR
pynlpir.open()


# 分词示例
def sep(text):
    # 分词
    segments = pynlpir.segment(text)
    print('分词中')
    # 去除停用词
    segments = [segment[0] for segment in segments if segment[0] not in stop_words]
    return ' '.join(segments)


# 对文本列进行处理
# data['NLPIR'] = data['QUERYLIST'].apply(lambda x: sep(str(x)))
# 关闭 NLPIR
# pynlpir.close()
# 将处理后的数据保存到新的 CSV 文件中
# data[['NLPIR']].to_csv(nlpir_path, index=False, encoding='GB18030')




def no_stop(text):
    # 分词
    segments = pynlpir.segment(text)
    print('分词中')
    # 正则化
    regulation = ['[', ']',  '.', '?', '!', "'", '"']
    segments = [segment[0] for segment in segments if segment[0] not in regulation]
    return ' '.join(segments)


# 对文本列进行处理
data['NLPIR'] = data['QUERYLIST'].apply(lambda x: no_stop(str(x)))

# 关闭 NLPIR
pynlpir.close()
# 将处理后的数据保存到新的 CSV 文件中
data[['NLPIR']].to_csv(NlpIr, index=False, encoding='GB18030')
