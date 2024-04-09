import gensim
import pandas as pd

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

jieba_train_path = '../1-process_data/jieba/jj.csv'
# 训练的语料

data = pd.read_csv(jieba_train_path, encoding='GB18030')
corpus_cn = data.head(50).values[:]
print(corpus_cn)

# 利用语料训练模型
model = Word2Vec(corpus_cn, window=5, min_count=1)

# 基于2d PCA拟合数据
# X = model[model.wv.vocab]
X = model.wv[model.wv.key_to_index]
pca = PCA(n_components=2)
# pca = PCA(n_components=3)
result = pca.fit_transform(X)

# 设置
pyplot.rcParams['font.sans-serif'] = ['SimHei']  # Use a Chinese font that supports the characters
pyplot.rcParams['axes.unicode_minus'] = False  # Ensure that minus signs are displayed correctly
# 可视化展示

# Create a 3D scatter plot
# fig = pyplot.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(result[:, 0], result[:, 1], result[:, 2])

pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.key_to_index)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

pyplot.show()
