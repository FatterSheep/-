"""
BoW是将文本转换为矢量最简单的方法,
使用他将句子或文档转换为矢量。

"""
import itertools

import jieba
import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import Embedding, Dense, LSTM, SpatialDropout1D
from keras.src.utils import to_categorical
from keras_preprocessing.text import Tokenizer
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
import keras.models
import keras.layers

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sympy.physics.units import cm

# min_df = 2，指的是至少有2个文档包含这个词条，才会保留
# max_df= 0.5，指的是词语文档频率高于max_df，则被过滤。
vectorizer = CountVectorizer()

df = pd.read_csv('../1-process_data/seperate/labels_all_process.csv', encoding='GB18030')

# 获取 JIEBA 列的前 n 行数据作为语料库
# corpus_cn = data['JIEBA'].head(25000).tolist()

'''
初始化数据
'''


jie = df['JIEBA'].tolist()
nlp = df['NLPIR'].tolist()
thu = df['THULAC'].tolist()
age = df['AGE'].tolist()
gender = df['GENDER'].tolist()
education = df['EDUCATION'].tolist()

df['merged_corpus'] = ((df['JIEBA'].astype(str) +
                       df['NLPIR'].astype(str) +
                       df['THULAC'].astype(str)) +
                       df['JIEBA_Length'].astype(str) +
                       df['NLPIR_Length'].astype(str) +
                       df['THULAC_Length'].astype(str))

merged_corpus = df['merged_corpus'].tolist()


# 初始化空集合来存储单词
vocabulary = set()

'''
创建词典
'''


def vocal(corpus):
    # 遍历文本数据，将单词添加到词汇表中
    # for text in corpus_cn:
    for text in corpus:
        words = str(text).split()  # 假设单词是通过空格分隔的
        vocabulary.update(words)

    # 查看词汇表大小
    print("Vocabulary size:", len(vocabulary))
    return vocabulary
    # 可选：将词汇表保存到文件
    # with open('vocabulary.txt', 'w', encoding='utf-8') as vocab_file:
    #     for word in vocabulary:
    #         vocab_file.write(word + '\n')


def bow(corpus):
    # 对所有关键词的frequency进行降片排序，只取前n个作为关键词
    bag_of_words_model_small = CountVectorizer(max_df=0.90)
    # 统计词频并转换为词袋模型
    bag_of_words_matrix = bag_of_words_model_small.fit_transform(corpus)
    return bag_of_words_matrix


"""
定义TopN词袋
"""


# 分词函数，使用 jieba 进行中文分词
def tokenize_text(corpus):
    tokenized_corpus = []
    for text in corpus:
        tokens = jieba.lcut(text)  # 使用 jieba 进行分词
        tokenized_corpus.append(tokens)
    return tokenized_corpus


def bow_top_n(corpus, n):
    # 返回每一行的dataframe返回前10个出现颜率最高词并以向最形式返回
    #:corpus: 输入文本语料库
    #:return:idataframeFNN
    tokenized_corpus = tokenize_text(corpus)  # 对文本进行分词处理
    # 对所有关键词的frequency进行降片排序，只取前n个作为关键词
    bag_of_words_model_small = CountVectorizer(max_features=n)
    # 统计词频并转换为dataframe
    bag_of_word_df_small = pd.DataFrame(bag_of_words_model_small.fit_transform(corpus).todense())
    # 给dataframe添加列名
    bag_of_word_df_small.columns = sorted(bag_of_words_model_small.vocabulary_)
    return bag_of_word_df_small


'''
创建词袋
'''
# df_3 = bow_top_n(corpus_cn, 3)
# print(df_3.head())
# df_3.to_csv('BW.csv', encoding='GB18030')
'''
筛选出Top20词袋
'''
# df_4 = bow_top_n(corpus_cn, 20)
# df_4.to_csv('BW_TOP.csv', encoding='GB18030')
# print(df_4.head())

'''
分类用的数据
'''


def bow_top(corpus, m, ag, gen, edu):
    # 返回每一行的 DataFrame，包含前 n 个出现频率最高的词作为特征，同时包含其他标签列
    # :corpus: 输入文本语料库
    # :n: 提取的前 n 个关键词
    # :age_labels: 年龄标签列表
    # :gender_labels: 性别标签列表
    # :education_labels: 受教育程度标签列表
    # 初始化 CountVectorizer，提取前 n 个关键词


    bag_of_words_model_small = CountVectorizer(min_df=5, max_features=m)
    # 提取关键词并生成 DataFrame
    bag_of_word_df_small = pd.DataFrame(bag_of_words_model_small.fit_transform(corpus).todense())
    # 设置列名为关键词
    bag_of_word_df_small.columns = sorted(bag_of_words_model_small.vocabulary_)

    # 添加其他标签列到 DataFrame
    bag_of_word_df_small['AGE'] = ag
    bag_of_word_df_small['GENDER'] = gen
    bag_of_word_df_small['EDUCATION'] = edu

    return bag_of_word_df_small


# # 提取前 10 个关键词及其他标签，并生成 DataFrame
# result_df = bow_top_n(corpus, 10, age_labels, gender_labels, education_labels)
# print(result_df)


'''
手肘法求k值
'''


def leg():
    # 手肘法求最佳K值
    distortions = []
    K = range(1, 10)
    X = bow_top_n(jie, 15000)
    # X = bow(corpus_cn)
    # X = vectorize_text(corpus_cn)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, n_init=10)
        kmeanModel.fit(X)
        # kmeanModel.fit_predict(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'),
                                      axis=1)) / X.shape[0])
    # 绘制时部图形
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title(' The Elbow Method showing the optimal number of clusters')
    plt.show()


# leg()

'''
K均值聚类
'''


def KM():
    X = bow_top_n(jie, 10000)
    # 设置要进行的聚类数量 K
    K = 3  # 这里假设要聚类为 3 类
    # 使用 K 均值算法进行聚类
    kmeans = KMeans(n_clusters=K, n_init=10)
    kmeans.fit(X)

    # 打印每个样本所属的聚类标签
    print("每个样本的聚类标签:")
    for i, label in enumerate(kmeans.labels_):
        print(f"样本 {i}: 属于聚类 {label}")


# KM()


'''
降维可视化处理
的K均值聚类
'''


def KM_plt():
    # 创建词袋模型特征
    # X = vectorize_text(corpus_cn)
    X = bow_top_n(jie, 1000)
    print(X)
    # 使用 TruncatedSVD 进行特征降维
    svd = TruncatedSVD(n_components=50)
    X_svd = svd.fit_transform(X)
    # 使用 t-SNE 进行降维，将高维特征降至二维
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X_svd)
    # 进行 KMeans 聚类，这里假设已经有了聚类标签 labels
    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit_predict(X_tsne)
    # 绘制散点图
    plt.figure(figsize=(20, 20))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', s=10)
    plt.legend(*scatter.legend_elements(), title='Clusters')
    plt.title('t-SNE Visualization of Text Clustering')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter)
    plt.show()


# KM_plt()


'''
混淆矩阵
'''


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


'''
朴素贝叶斯分类
'''


def Multinomial():
    #  标记好的文本数据和对应的标签
    # JIEBA = data['JIEBA'].head(10)
    # print(JIEBA)

    df['merged_corpus'] = (df['JIEBA'].astype(str) + df['JIEBA_Length'].astype(str))
    X = df['merged_corpus'].tolist()
    y = df['EDUCATION'].tolist()

    # x = bow_top(merged_corpus, 10000, age, education, gender)
    #
    # X = x.iloc[:, :-3]
    # y = x['EDUCATION'].values

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, random_state=0)

    # # 统计不同类别的数量，用于计算每个类别的先验概率
    # class_counts = pd.Series(y_train).value_counts(normalize=True)
    # # 计算每个类别的先验概率，可以根据实际情况调整类别的权重
    # class_priors = [class_counts[1], class_counts[2]]  # 示例中取了前两个类别的先验概率

    # 使用 CountVectorizer 对文本数据进行向量化处理
    vectorizer = CountVectorizer(max_df=0.90)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    # X_log = np.log1p(X_train_vectorized)  # 对词频值应用对数变换
    # X_test_vectorized = vectorizer.transform(X_test)

    # 使用朴素贝叶斯分类器建立模型并进行训练
    clf = MultinomialNB(alpha=10)
    clf.fit(X_train_vectorized, y_train)

    # 在测试集上评估模型性能
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print(f"朴素贝叶斯的准确率：{accuracy}")
    print('朴素贝叶斯')
    cnf_matrix = confusion_matrix(y_test, y_pred)

    print("Recall metric in the testing dataset:", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
    print("accuracy metric in the testing dataset: ",
          (cnf_matrix[1, 1] + cnf_matrix[0, 0]) /
          (cnf_matrix[0, 0] + cnf_matrix[1, 1] + cnf_matrix[0, 1] + cnf_matrix[1, 0]))

    # Plot non normalized confusion matrix
    class_names = [0, 1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
    plt.show()


# Multinomial()

'''
逻辑回归
'''


def Logistic():
    # X = bow_top(corpus_cn, 10000, age_labels, gender_labels, education_labels)

    # x = bow_top(merged_corpus, 10000, age, gender, education)
    # LR_model = LogisticRegression(penalty='l2', C=1.0)

    # X = x.iloc[:, :-3]
    # y = x['EDUCATION'].values
    # 假设 JIEBA_Length 是需要进行权重分配的列，将其进行缩放
    scaler = MinMaxScaler()
    df['JIEBA_Length_Scaled'] = scaler.fit_transform(df[['JIEBA_Length']])
    df['merged_corpus'] = (df['JIEBA'].astype(str) + df['JIEBA_Length_Scaled'].astype(str))
    X = df['merged_corpus'].tolist()
    X = np.array(X)
    y = df['EDUCATION'].tolist()
    y = np.array(y)

    # 定义 CountVectorizer 和逻辑回归模型
    vectorizer = CountVectorizer(max_df=0.90)  # 根据需要添加 CountVectorizer 的参数
    logistic_model = LogisticRegression(C=3.0)  # 根据需要添加逻辑回归的参数

    # 初始化 KFold
    k_fold = KFold(n_splits=3, shuffle=True, random_state=42)  # 5-fold 交叉验证，根据需要调整参数

    accuracies = []

    # 执行 KFold 交叉验证
    for train_idx, test_idx in k_fold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 向量化文本特征
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)

        # 训练逻辑回归模型
        logistic_model.fit(X_train_vectorized, y_train)

        # 预测并计算准确率
        accuracy = logistic_model.score(X_test_vectorized, y_test)
        accuracies.append(accuracy)

    # 打印准确率
    print("Accuracies:", accuracies)
    print("Mean Accuracy:", np.mean(accuracies))

# Logistic()


def Net():
    # 假设我们有一个包含文本和标签的数据集
    X = bow_top(jie, 10, age, gender, education)
    print(X)
    X_T = X.iloc[:, :-3]
    print(len(X_T))
    X_A = X['GENDER'].values
    print(len(X_A))

    # # 使用Tokenizer对文本进行向量化
    # max_words = 1000
    # tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    # tokenizer.fit_on_texts(X_T)
    # sequences = tokenizer.texts_to_sequences(X_T)
    #
    # # 对文本序列进行填充
    # maxlen = 20
    # X = pad_sequences(sequences, maxlen=maxlen)
    #
    # # 将标签转换为numpy数组
    # y = np.array(X_A)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_T, X_A, test_size=0.2, random_state=42)
    max_words = 1000
    maxlen = 20
    # 构建神经网络模型
    model = Sequential()
    model.add(Embedding(max_words, 50, input_length=maxlen))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 训练模型
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"神经网络的准确率：{accuracy}")


# Net()

'''
def TT(num_classes, column):
    model = Sequential()
    vocal()
    model.add(Embedding(input_dim=len(vocabulary), output_dim=100, input_length=10000))
    model.add(LSTM(units=100))
    model.add(Dense(num_classes, activation='softmax'))  # 这里的num_classes指的是标签类别有几个，类似于Age为2个

    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 数据准备
    # X = bow_top(corpus_cn, 10000, age_labels, gender_labels, education_labels)
    X = bow_top(jie, 10000, age, gender, education)
    X_T = X.iloc[:, :-3]
    X_A = X[column].values
    X_train, X_test, y_train, y_test = train_test_split(X_T, X_A,
                                                        test_size=0.2, random_state=42)
    # 划分验证集（这里以20%的数据作为验证集）
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 将标签数据转换为独热编码
    y_train_encoded = to_categorical(y_train, num_classes=num_classes)
    y_val_encoded = to_categorical(y_val, num_classes=num_classes)

    # 训练模型
    # model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))
    model.fit(X_train, y_train)
    # 模型评估
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy}')
    # 模型预测
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率：{accuracy}")

# TT(2, 'GENDER')
'''


def make_Silhouette_plot(X, n_clusters):
    plt.xlim([-0.1, 1])
    plt.ylim([0, len(X) + (n_clusters + 1) * 10])
    # 建立聚类模型
    clusterer = KMeans(n_clusters=n_clusters,
                       max_iter=1000,
                       n_init=10,
                       init="k-means++",
                       random_state=10)

    # 聚类预测生成标签label
    cluster_label = clusterer.fit_predict(X)
    # 计算轮廓系数均值（整体数据样本）
    silhouette_avg = silhouette_score(X, cluster_label)
    print(f"n_clusterers: {n_clusters}, silhouette_score_avg:{silhouette_avg}")

    # 单个数据样本
    sample_silhouette_value = silhouette_samples(X, cluster_label)
    y_lower = 10

    for i in range(n_clusters):
        # 第i个簇群的轮廓系数
        i_cluster_silhouette_value = sample_silhouette_value[cluster_label == i]
        # 进行排序
        i_cluster_silhouette_value.sort()
        size_cluster_i = i_cluster_silhouette_value.shape[0]
        y_upper = y_lower + size_cluster_i
        # 颜色设置
        color = cm.nipy_spectral(float(i) / n_clusters)

        # 边界填充
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            i_cluster_silhouette_value,
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )
        # 添加文本信息
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
        plt.title(f"The Silhouette Plot for n_cluster = {n_clusters}", fontsize=26)
        plt.xlabel("The silhouette coefficient values", fontsize=24)
        plt.ylabel("Cluter Label", fontsize=24)
        plt.axvline(x=silhouette_avg, color="red", linestyle="--")
        # x-y轴的刻度标签
        plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.yticks([])

#
# range_n_clusters = list(range(2, 10))
#
# for n in range_n_clusters:
#     print(f"N cluster:{n}")
#     make_Silhouette_plot(data, n)
#     plt.savefig(f"Silhouette_Plot_{n}.png")
#     plt.close()


def Stracking():
    # 我们有一个包含文本和标签的数据集
    # tfidf_vectorizer = CountVectorizer(max_df=0.90)
    # X = tfidf_vectorizer.fit_transform(jie)  # 将文本转换为词频矩阵
    # y = df['EDUCATION'].values

    scaler = MinMaxScaler()
    df['JIEBA_Length_Scaled'] = scaler.fit_transform(df[['JIEBA_Length']])
    df['merged_corpus'] = (df['JIEBA'].astype(str) + df['JIEBA_Length_Scaled'].astype(str))
    X = df['merged_corpus'].tolist()  # 600
    X = np.array(X)
    y = df['EDUCATION'].tolist()  # 600
    y = np.array(y)

    # 定义 CountVectorizer 和逻辑回归模型
    vectorizer = CountVectorizer(max_df=0.90)  # 根据需要添加 CountVectorizer 的参数

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # 使用 CountVectorizer 来构建词袋模型，并只使用前 10,000 个特征词
    # vectorizer = CountVectorizer(max_df=0.90)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # 创建基础模型
    svm = LinearSVC()
    nb = MultinomialNB(alpha=10)
    lr = LogisticRegression(C=3.0)

    # 定义权重（示例：将类别1的权重设为2，类别0的权重设为1）
    class_weights = {0: 1, 1: 2}

    # 生成元特征
    kf = KFold(n_splits=5)
    meta_features = None

    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]

        svm.fit(X_train_fold, y_train_fold)
        nb.fit(X_train_fold, y_train_fold)
        lr.fit(X_train_fold, y_train_fold)

        pred_svm = svm.predict(X_val_fold)
        pred_nb = nb.predict(X_val_fold)
        pred_lr = lr.predict(X_val_fold)

        meta_fold = np.column_stack((pred_svm, pred_nb, pred_lr))  # 将三个基础模型的预测结果堆叠在一起


        if meta_features is None:
            meta_features = meta_fold
        else:
            meta_features = np.vstack((meta_features, meta_fold))

    # 使用元特征训练次级模型
    svm_meta = LinearSVC()
    svm_meta.fit(meta_features, y_train)  # 使用元特征训练次级模型

    # 在测试集上生成元特征并进行预测
    pred_svm_test = svm.predict(X_test)
    pred_nb_test = nb.predict(X_test)
    pred_lr_test = lr.predict(X_test)

    meta_test = np.column_stack((pred_svm_test, pred_nb_test, pred_lr_test))
    predictions = svm_meta.predict(meta_test)  # 使用次级模型进行最终的预测

    # 评估模型
    accuracy = accuracy_score(y_test, predictions)
    print('Stacking')
    print(accuracy)

    # cnf_matrix = confusion_matrix(y_test, predictions)
    #
    # print("Recall metric in the testing dataset:", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
    # print("accuracy metric in the testing dataset: ",
    #       (cnf_matrix[1, 1] + cnf_matrix[0, 0]) /
    #       (cnf_matrix[0, 0] + cnf_matrix[1, 1] + cnf_matrix[0, 1] + cnf_matrix[1, 0]))
    #
    # # Plot non normalized confusion matrix
    # class_names = [0, 1]
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
    # plt.show()

'''
    # 基础模型
    models = [
        RandomForestClassifier(random_state=42),
        GradientBoostingClassifier(random_state=42)
    ]

    # 次级模型
    meta_model = LogisticRegression()

    # 用于在交叉验证中创建次级训练集的类
    class StackingModel(BaseEstimator, TransformerMixin):
        def __init__(self, models, meta_model):
            self.models = models
            self.meta_model = meta_model

        def fit(self, X, y):
            self.meta_features = np.zeros((X.shape[0], len(self.models)))
            skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
            for i, model in enumerate(self.models):
                for train_index, val_index in skf.split(X, y):
                    model.fit(X[train_index], y[train_index])
                    self.meta_features[val_index, i] = model.predict(X[val_index])
            self.meta_model.fit(self.meta_features, y)
            return self

        def predict(self, X):
            test_meta_features = np.column_stack([
                model.predict(X) for model in self.models
            ])
            return self.meta_model.predict(test_meta_features)

    # 创建并训练 Stacking 模型
    stacking_model = StackingModel(models, meta_model)
    stacking_model.fit(X_train, y_train)

    # 预测并评估
    y_pred = stacking_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
'''

Stracking()
