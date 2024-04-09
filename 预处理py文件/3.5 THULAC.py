import pandas as pd
import thulac

# 分词示例
trainname = '../Train/Process_Query.csv'
stopword_path = '../stopwords/cn_stopwords.txt'
thulac_path = '../seperate/THULAC.csv'
ThuLac = '../no_stop/THULAC.csv'

data = pd.read_csv(trainname, encoding='GB18030')
# 停用词列表示例
with open(stopword_path, encoding='GB18030') as f:  # 可根据需要打开停用词库，然后加上不想显示的词语
    con = f.readlines()
    stop_words = set()  # 集合可以去重
    for i in con:
        i = i.replace("\n", "")  # 去掉读取每一行数据的\n
        stop_words.add(i)

# 初始化 THULAC
thu = thulac.thulac(seg_only=True)  # 仅分词，不进行词性标注


# 分词示例
def sep(text):
    # 分词
    segments = thu.cut(text, text=True)  # text=True 表示输出文本形式的分词结果
    print('分词中')
    # 去除停用词
    segments = [segment[0] for segment in segments if segment[0] not in stop_words]
    return ' '.join(segments)


# 对文本列进行处理
# data['THULAC'] = data['QUERYLIST'].apply(lambda x: sep(str(x)))
# 将处理后的数据保存到新的 CSV 文件中
# data[['THULAC']].to_csv(thulac_path, index=False, encoding='GB18030')



def no_stop(text):
    # 分词
    segments = thu.cut(text, text=True)  # text=True 表示输出文本形式的分词结果
    print('分词中')
    # 去除停用词
    regulation = ['[', ']',  '.', '?', '!', "'", '"']
    segments = [segment[0] for segment in segments if segment[0] not in regulation]
    return ' '.join(segments)


# 对文本列进行处理
data['THULAC'] = data['QUERYLIST'].apply(lambda x: no_stop(str(x)))
# 将处理后的数据保存到新的 CSV 文件中
data[['THULAC']].to_csv(ThuLac, index=False, encoding='GB18030')
