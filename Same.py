import codecs
import random

import gensim
import jieba
import pandas as pd
from gensim.models.word2vec import LineSentence, Word2Vec

data = pd.read_csv('../1-process_data/label/labels_all.csv', encoding='GB18030')
jie = data['JIEBA'].tolist()
# 对中文文本进行分词处理
# tokenized_data = [list(jieba.cut(text)) for text in jie]
'''
jie = df['JIEBA'].tolist()
nlp = df['NLPIR'].tolist()
thu = df['THULAC'].tolist()
df['merged_corpus'] = (df['JIEBA'].astype(str) +
                       df['NLPIR'].astype(str) +
                       df['THULAC'].astype(str))
merged_data = df['merged_corpus'].tolist()
# 创建包含要保存的列数据的新 DataFrame
data_to_save = pd.DataFrame({'merged_corpus': merged_data})
# 保存为 CSV 文件
data_to_save.to_csv('merged_corpus.csv', index=False, encoding='GB18030')
'''

# # 初始化并训练 Word2Vec 模型
# model = Word2Vec(tokenized_data, vector_size=100, window=5, min_count=5, workers=4)
# # 保存模型
# model.save("model.bin")

# # 加载预训练的Word2Vec模型
model = Word2Vec.load("model.bin")

# data = pd.read_csv('../1-process_data/Train/JIEBA.csv', encoding='GB18030')

# 随机选择10000行数据
random_indices = random.sample(range(len(data)), 10000)
# 对选定行进行同义词替换
for i in random_indices:
    text = data.at[i, 'JIEBA']  # 假设文本数据在 'JIEBA' 列中
    tokens = text.split()
    for j, token in enumerate(tokens):
        try:
            similar_words = model.wv.most_similar(token, topn=1)
            similar_word = similar_words[0][0]
            tokens[j] = similar_word
        except KeyError:
            continue
    replaced_text = ' '.join([word[0] for word in tokens])
    data.at[i, 'JIEBA'] = replaced_text  # 更新替换后的文本数据
# 保存替换后的数据
data.to_csv("use_in_classify.csv", index=False, encoding='GB18030')

# selected_data = data.iloc[random_indices]
# selected_data.at[i, 'JIEBA'] = replaced_text  # 更新替换后的文本数据
# for i, text in selected_data.iterrows():
#     tokens = text['JIEBA'].split()  # 假设文本数据在 'text_column' 列中
'''
# 对每行文本进行同义词替换
for i, text in enumerate(jb):
    # 将每行文本拆分成词语列表
    tokens = list(jieba.cut(text))
    token_data = [[word] for word in tokens]

    # 对列表中的每个词语进行同义词替换
    for j, sublist in enumerate(token_data):
        for k, word in enumerate(sublist):
            try:
                similar_words = model.wv.most_similar(word, topn=1)
                similar_word = similar_words[0][0]
                token_data[j][k] = similar_word
            except KeyError:
                continue

    # 将替换后的词语列表重新组合成文本
    replaced_text = ' '.join([word[0] for word in token_data])

    # 将替换后的文本更新回 DataFrame 中的对应位置
    data.at[i, 'JIEBA'] = replaced_text

# 将替换后的文本数据保存回 CSV 文件中
data.to_csv("use_in_classify.csv", index=False, encoding='GB18030')
'''

'''
# 假设你有一段文本
text = "受欢迎小狗排行榜"
# 将文本拆分成词语列表
tokens = text.split()
token_data = [list(jieba.cut(text)) for text in tokens]
split_data = [[word] for sublist in token_data for word in sublist]
print("替换前文本：", split_data)
# 对列表中的每个词语进行同义词替换
for i, text in enumerate(jie):
    tokens = text.split()
for i, sublist in enumerate(split_data):
    for j, word in enumerate(sublist):
        try:
            similar_words = model.wv.most_similar(word, topn=1)
            similar_word = similar_words[0][0]
            split_data[i][j] = similar_word
        except KeyError:
            continue
print("替换后文本：", split_data)
'''

# 查找近义词
# similar_words = model.wv.most_similar('女孩', topn=8)
# print(similar_words)


