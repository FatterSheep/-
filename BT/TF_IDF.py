import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC


df = pd.read_csv('../1-process_data/seperate/labels_all.csv', encoding='GB18030')

jie = df['JIEBA'].tolist()
nlp = df['NLPIR'].tolist()
thu = df['THULAC'].tolist()
age = df['AGE'].tolist()
gender = df['GENDER'].tolist()
education = df['EDUCATION'].tolist()

df['merged_corpus'] = (df['JIEBA'].astype(str) +
                       df['NLPIR'].astype(str) +
                       df['THULAC'].astype(str))

merged_data = df['merged_corpus'].tolist()

"""
定义TopN TF-IDF模型
"""


def Tf_Idf(corpus, n):
    # TF-IDF
    # 对所有关键词的frequency进行降片排序，只取前n个作为关键词
    tfidf_vectorizer = TfidfVectorizer(max_features=n)
    X_tfidf_model = tfidf_vectorizer.fit_transform(corpus)
    # 统计词频并转换为dataframe
    X_tfidf_df = pd.DataFrame(X_tfidf_model.todense(),
                              columns=tfidf_vectorizer.get_feature_names_out())  # 给dataframe添加列名
    return X_tfidf_df


# Tf_Idf(corpus_cn, 10)

def bow_top_n(corpus, n):
    # 返回每一行的dataframe返回前10个出现颜率最高词并以向最形式返回
    #:corpus: 输入文本语料库
    #:return:idataframeFNN

    # 对所有关键词的frequency进行降片排序，只取前n个作为关键词
    bag_of_words_model_small = TfidfVectorizer(max_features=n)
    # 统计词频并转换为dataframe
    bag_of_word_df_small = pd.DataFrame(bag_of_words_model_small.fit_transform(corpus).todense())
    # 给dataframe添加列名
    bag_of_word_df_small.columns = sorted(bag_of_words_model_small.vocabulary_)
    return bag_of_word_df_small


'''
筛选出Top20词袋
'''
# df_4 = bow_top_n(corpus_cn, 20)
# df_4.to_csv('../1-2-keyword/TF_TOP.csv', encoding='GB18030')
# print(df_4.head())


'''
分类用的数据
'''


def TFIDF_top(corpus, n, age_labels, gender_labels, education_labels):
    # 返回每一行的 DataFrame，包含前 n 个出现频率最高的词作为特征，同时包含其他标签列
    # :corpus: 输入文本语料库
    # :n: 提取的前 n 个关键词
    # :age_labels: 年龄标签列表
    # :gender_labels: 性别标签列表
    # :education_labels: 受教育程度标签列表

    # 初始化 CountVectorizer，提取前 n 个关键词
    tfidf_vectorizer = TfidfVectorizer(max_features=n)
    X_tfidf_model = tfidf_vectorizer.fit_transform(corpus)
    # 提取关键词并生成 DataFrame
    X_tfidf_df = pd.DataFrame(X_tfidf_model.todense(),
                              columns=tfidf_vectorizer.get_feature_names_out())
    # 添加其他标签列到 DataFrame
    X_tfidf_df['AGE'] = age_labels
    X_tfidf_df['GENDER'] = gender_labels
    X_tfidf_df['EDUCATION'] = education_labels

    return X_tfidf_df


def leg():
    # 手肘法求最佳K值
    distortions = []
    K = range(1, 10)
    X = Tf_Idf(jie, 15000)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, n_init=10)
        kmeanModel.fit(X)
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
    X = Tf_Idf(jie, 10000)
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
    X = Tf_Idf(jie, 10000)
    # 使用 TruncatedSVD 进行特征降维
    svd = TruncatedSVD(n_components=50)
    X_svd = svd.fit_transform(X)
    # 使用 t-SNE 进行降维，将高维特征降至二维
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X_svd)
    # 进行 KMeans 聚类，这里假设已经有了聚类标签 labels
    kmeans = KMeans(n_clusters=5)
    labels = kmeans.fit_predict(X_tsne)
    # 绘制散点图
    plt.figure(figsize=(8, 6))
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
文本分类
'''


def Multinomial():
    #  标记好的文本数据和对应的标签
    # JIEBA = data['JIEBA'].head(10)
    # print(JIEBA)
    # 使用 CountVectorizer 进行特征提取
    # vectorizer = TfidfVectorizer()

    # X = TFIDF_top(merged_data, 400, age, gender, education)
    # # X_A = X.iloc[:, -3]
    # X_A = X['AGE'].values
    # # X_T = X.iloc[:, :-3]
    # X_T = X.iloc[:, :-3]
    # # X_J = vectorizer.fit_transform(X_T)
    # X = vectorizer.fit_transform(JieBa)

    df['merged_corpus'] = (df['JIEBA'].astype(str) + df['JIEBA_Length'].astype(str))
    X = df['merged_corpus'].values
    y = df['AGE'].values
    # 使用 TfidfVectorizer 对文本数据进行向量化处理
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9)
    X_train_vectorized = tfidf_vectorizer.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_train_vectorized, y,
                                                        test_size=0.2, random_state=42)

    # 使用朴素贝叶斯分类器建立模型并进行训练
    clf = MultinomialNB(alpha=10)
    clf.fit(X_train, y_train)

    # 在测试集上评估模型性能
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率：{accuracy}")


# Multinomial()


def Logistic():
    # X = TFIDF_top(merged_data, 200, age, gender, education)


    df['merged_corpus'] = (df['JIEBA'].astype(str) + df['JIEBA_Length'].astype(str))
    X = df['merged_corpus'].values  # (250)
    y = df['AGE'].values
    # 使用 TfidfVectorizer 对文本数据进行向量化处理
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9)
    X_train_vectorized = tfidf_vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_train_vectorized, y,
                                                        test_size=0.2, random_state=42)
    LR_model = LogisticRegression(C=3.0)
    LR_model = LR_model.fit(X_train, y_train)
    y_pred = LR_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率：{accuracy}")


Logistic()



def Stracking():
    # 假设我们有一个包含文本和标签的数据集
    # # 使用 CountVectorizer 来构建词袋模型，并只使用前 10,000 个特征词
    # tfidf_vectorizer = TfidfVectorizer(max_features=600)
    scaler = MinMaxScaler()
    df['JIEBA_Length_Scaled'] = scaler.fit_transform(df[['JIEBA_Length']])
    df['merged_corpus'] = (df['JIEBA'].astype(str) + df['JIEBA_Length_Scaled'].astype(str))
    X = df['merged_corpus'].tolist()
    X = np.array(X)
    y = df['AGE'].tolist()
    y = np.array(y)

    # 定义 CountVectorizer 和逻辑回归模型
    tfidf_vectorizer = TfidfVectorizer(max_df=0.85,min_df=3)  # 根据需要添加 CountVectorizer 的参数


    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)

    # 创建基础模型
    svm = LinearSVC()
    nb = MultinomialNB(alpha=10)
    lr = LogisticRegression(C=3.0)

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

# Stracking()
