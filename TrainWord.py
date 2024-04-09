# coding=utf-8
import gzip
import gensim

from gensim.test.utils import common_texts
# size：詞向量的大小，window：考慮上下文各自的長度
# min_count：單字至少出現的次數，workers：執行緒個數
model_simple = gensim.models.Word2Vec(sentences=common_texts, window=1,
                                      min_count=1, workers=4)
# 傳回 有效的字數及總處理字數
print(model_simple.train([["hello", "world", "michael"]], total_examples=1, epochs=2))

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]

model_simple = gensim.models.Word2Vec(min_count=1)
model_simple.build_vocab(sentences)  # 建立生字表(vocabulary)
print(model_simple.train(sentences, total_examples=model_simple.corpus_count
                         , epochs=model_simple.epochs))


# 載入 OpinRank 語料庫：關於車輛與旅館的評論
data_file="../nlp-in-practice-master/word2vec/reviews_data.txt.gz"

with gzip.open (data_file, 'rb') as f:
    for i,line in enumerate (f):
        print(line)
        break


# 讀取 OpinRank 語料庫，並作前置處理
def read_input(input_file):
    with gzip.open (input_file, 'rb') as f:
        for i, line in enumerate (f):
            # 前置處理
            yield gensim.utils.simple_preprocess(line)

# 載入 OpinRank 語料庫，分詞
documents = list(read_input(data_file))
# print(documents)


print(len(documents))

# Word2Vec 模型訓練，約10分鐘
model = gensim.models.Word2Vec(documents,
                               vector_size=150, window=10,
                               min_count=2, workers=10)
print(model.train(documents, total_examples=len(documents), epochs=10))


# 測試『骯髒』相似詞
w1 = "dirty"
print(model.wv.most_similar(positive=w1))
# positive：相似詞


# 測試『禮貌』相似詞
w1 = ["polite"]
print(model.wv.most_similar(positive=w1, topn=6))
# topn：只列出前 n 名


# 測試『法國』相似詞
w1 = ["france"]
print(model.wv.most_similar(positive=w1, topn=6))
# topn：只列出前 n 名


# 測試『床、床單、枕頭』相似詞及『長椅』相反詞
w1 = ["bed",'sheet','pillow']
w2 = ['couch']
print(model.wv.most_similar(positive=w1, negative=w2, topn=10))
# negative：相反詞

# 比較兩詞相似機率
print(model.wv.similarity(w1="dirty", w2="smelly"))
print(model.wv.similarity(w1="dirty", w2="dirty"))

print(model.wv.similarity(w1="dirty", w2="clean"))

# 選出較不相似的字詞
print(model.wv.doesnt_match(["cat", "dog", "france"]))

