import multiprocessing
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models.word2vec import Word2Vec
# from tensorflow.python.ops import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

warnings.filterwarnings("ignore")  # 忽略警告信息
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device =', device)
train_data = pd.read_csv('../1-process_data/seperate/labels_all.csv',
                         encoding='GB18030')
train_data = train_data.head(15000)
print('STEP.1', '-' * 19)
''' 加载自定义中文数据 '''
print(train_data.columns)


# 构造数据集迭代器
def coustom_data_iter(texts, labels):
    for x, y in zip(texts, labels):
        yield x, y


x = train_data['JIEBA'].values[:]
# x = train_data['JIEBA'].tolist()
# 多类标签的one-hot展开
y = train_data['EDUCATION'].values[:]
# y = train_data['GENDER'].tolist()

'''调用gensim库
步骤：
第一步构建一个空模型
第二步使用 build_vocab 方法根据输入的文本数据 x 构建词典。
build_vocab 方法会统计输入文本中每个词汇出现的次数，并按照词频从高到低的顺序将词汇加入词典中。
第三步使用 train 方法对模型进行训练，
total_examples 参数指定了训练时使用的文本数量，
这里使用的是 w2v.corpus_count 属性，表示输入文本的数量
'''

print('STEP.2', '-' * 19)
# 训练 Word2Vec 浅层神经网络模型
w2v = Word2Vec(vector_size=100,  # 是指特征向量的维度，默认为100。
               min_count=5)  # 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5。

w2v.build_vocab(x)
w2v.train(x,
          total_examples=w2v.corpus_count,
          epochs=20)

'''
定义了一个函数 average_vec(text)，它接受一个包含多个词的列表 text 作为输入，
并返回这些词对应词向量的平均值。该函数:
1.首先初始化一个形状为 (1, 100) 的全零 numpy 数组来表示平均向量
2.然后遍历 text 中的每个词，并尝试从 Word2Vec 模型 w2v 中使用 wv 属性获取其对应的词向量。
如果在模型中找到了该词，函数将其向量加到 vec 中。如果未找到该词，函数会继续迭代下一个词
3.最后，函数返回平均向量 vec

使用列表推导式将 average_vec() 函数应用于列表 x 中的每个元素。
得到的平均向量列表使用 np.concatenate() 连接成一个 numpy 数组 x_vec，
该数组表示 x 中所有元素的平均向量。x_vec 的形状为 (n, 100)，其中 n 是 x 中元素的数量。
'''


# 将文本转化为向量
def average_vec(text):
    vec = np.zeros(100).reshape((1, 100))
    for word in text:
        try:
            vec += w2v.wv[word].reshape((1, 100))
        except KeyError:
            continue
    return vec


# 将词向量保存为 Ndarray
x_vec = np.concatenate([average_vec(z) for z in x])

# 保存 Word2Vec 模型及词向量
w2v.save('w2v_model.pkl')

train_iter = coustom_data_iter(x_vec, y)
print(len(x), len(x_vec)) # 88340 88340

print('STEP.3', '-' * 19)
''' 准备数据处理管道 '''
# label_name = list(set(train_data['GENDER'].values[:]))
label_name = list(set(train_data['GENDER'].values[:]))
# label_name = train_data['GENDER'].tolist()
# label_name = list(set(train_data['GENDER'].tolist()))
print(label_name) # [1, 2]

# 生成数据批次和迭代器
text_pipeline = lambda x: average_vec(x)
label_pipeline = lambda x: label_name.index(x)
print(text_pipeline("你在干嘛")) # 把这句话变成词向量表示
print(label_pipeline(1)) # 转换为标签索引，1是原始标签值，索引为0

''' 生成数据批次和迭代器 '''


def collate_batch(batch):
    label_list, text_list = [], []

    for (_text, _label) in batch:
        # 标签列表
        label_list.append(label_pipeline(_label))

        # 文本列表
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.float32)
        text_list.append(processed_text)

    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.cat(text_list)

    return text_list.to(device), label_list.to(device)


print('STEP.4', '-' * 19)
''' 搭建文本分类模型 '''


# 搭建模型
class TextClassificationModel(nn.Module):

    def __init__(self, num_class):
        super(TextClassificationModel, self).__init__()
        self.fc = nn.Linear(100, num_class)

    def forward(self, text):
        return self.fc(text)


print('STEP.5 Initialize', '-' * 19)
# 初始化模型

num_class = len(label_name)
vocab_size = 100000
em_size = 12
model = TextClassificationModel(num_class).to(device)

''' 训练函数 '''


# 定义训练和评估函数


def train(dataloader):
    model.train()  # 切换为训练模式
    total_acc, train_loss, total_count = 0, 0, 0
    log_interval = 50
    start_time = time.time()

    for idx, (text, label) in enumerate(dataloader):
        predicted_label = model(text)

        optimizer.zero_grad()  # grad属性归零
        loss = criterion(predicted_label, label)  # 计算网络输出和真实值之间的差距，label为真实值
        loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # 梯度裁剪
        optimizer.step()  # 每一步自动更新

        # 记录acc与loss
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        train_loss += loss.item()
        total_count += label.size(0)

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:1d} | {:4d}/{:4d} batches '
                  '| train_acc {:4.3f} train_loss {:4.5f}'.format(epoch, idx, len(dataloader),
                                                                  total_acc / total_count, train_loss / total_count))
            total_acc, train_loss, total_count = 0, 0, 0
            start_time = time.time()


''' 评估函数 '''


def evaluate(dataloader):
    model.eval()  # 切换为测试模式
    total_acc, train_loss, total_count = 0, 0, 0

    with torch.no_grad():
        for idx, (text, label) in enumerate(dataloader):
            predicted_label = model(text)

            loss = criterion(predicted_label, label)  # 计算loss值
            # 记录测试数据
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            train_loss += loss.item()
            total_count += label.size(0)

    return total_acc / total_count, train_loss / total_count


# 训练模型

# 拆分数据集并运行模型
if __name__ == '__main__':
    # 超参数
    EPOCHS = 10  # epoch
    LR = 5  # 学习率
    BATCH_SIZE = 64  # batch size for training

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None
    # 构建数据集
    train_iter = coustom_data_iter(train_data['JIEBA'].values[:], train_data['GENDER'].values[:])
    # train_iter = coustom_data_iter(train_data['JIEBA'].tolist(), train_data['GENDER'].tolist())
    # train_dataset = to_map_style_dataset(train_iter)
    train_dataset = list(train_iter)
    # 划分数据集
    num_train = int(len(train_dataset) * 0.8)
    split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])
    # 加载数据集
    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_batch)  # shuffle表示随机打乱
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

# split_train_, split_valid_ = random_split(train_dataset,
#                                           [int(len(train_dataset) * 0.8), int(len(train_dataset) * 0.2)])
#
# train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
#                               shuffle=True, collate_fn=collate_batch)
#
# valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
#                               shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    # val_acc, val_loss = evaluate(valid_dataloader)
    accu_val, loss_val = evaluate(valid_dataloader)

    # 获取当前的学习率
    lr = optimizer.state_dict()['param_groups'][0]['lr']

    # if total_accu is not None and total_accu > val_acc:
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        # total_accu = val_acc
        total_accu = accu_val
    print('-' * 69)
    # print('| epoch {:1d} | time: {:4.2f}s | '
    #       'valid_acc {:4.3f} valid_loss {:4.3f} | lr {:4.6f}'.format(epoch,
    #                                                                  time.time() - epoch_start_time,
    #                                                                  val_acc, val_loss, lr))
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid_acc {:8.3f} valid_loss {:8.3f} | lr {:8.6f}'.format(epoch, time.time() - epoch_start_time, accu_val,
                                                                     loss_val, lr))
    print('-' * 69)
    torch.save(model.state_dict(), 'model_TextClassification.pth')

    print('Checking the results of test dataset.')
    accu_test, loss_test = evaluate(valid_dataloader)
    print('test accuracy {:8.3f}, test loss {:8.3f}'.format(accu_test, loss_test))

    # test_acc, test_loss = evaluate(valid_dataloader)
    # print('模型准确率为：{:5.4f}'.format(test_acc))

''' 预测函数 '''


def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text), dtype=torch.float32)
        print(text.shape)
        output = model(text)
        return output.argmax(1).item()


''' 以下是预测 '''
if __name__ == '__main__':
    model.load_state_dict(torch.load('model_TextClassification.pth'))
    ex_text_str = "汽车"
    model = model.to("cpu")
    print("该文本的类别是：%s" % label_name[predict(ex_text_str, text_pipeline)])
