import pandas as pd

food = 'similar_words.txt'
pe = 'PE.txt'
hosp = 'hospital.txt'
elect = 'elect.txt'
game = 'game.txt'
trip = 'trip.txt'
video = 'video.txt'

# 读取txt词典文件内容并存储到列表中
with open(food, 'r', encoding='utf-8') as file:
    dictionary = [line.strip() for line in file]

with open(pe, 'r', encoding='utf-8') as file:
    pe = [line.strip() for line in file]

with open(hosp, 'r', encoding='utf-8') as file:
    hosp = [line.strip() for line in file]

with open(elect, 'r', encoding='utf-8') as file:
    elect = [line.strip() for line in file]

with open(game, 'r', encoding='utf-8') as file:
    game = [line.strip() for line in file]

with open(trip, 'r', encoding='utf-8') as file:
    trip = [line.strip() for line in file]

with open(video, 'r', encoding='utf-8') as file:
    video = [line.strip() for line in file]

# 读取CSV文件
df = pd.read_csv('../1-process_data/seperate/labels_all.csv', encoding='GB18030')  # 替换为实际的CSV文件名

# 在CSV文件中检查JIEBA列中的词汇是否在词典中，并添加新列标记为1
df['美食'] = 0  # 创建新列并初始化为0
for index, row in df.iterrows():
    if any(word in dictionary for word in str(row['JIEBA']).split()):
        df.at[index, '美食'] = 1  # 将匹配到词典的行设置为美食

df['体育'] = 0  # 创建新列并初始化为0
for index, row in df.iterrows():
    if any(word in pe for word in str(row['JIEBA']).split()):
        df.at[index, '体育'] = 1  # 将匹配到词典的行设置为体育

df['医疗'] = 0  # 创建新列并初始化为0
for index, row in df.iterrows():
    if any(word in hosp for word in str(row['JIEBA']).split()):
        df.at[index, '医疗'] = 1  # 将匹配到词典的行设置为医疗

df['电子产品'] = 0  # 创建新列并初始化为0
for index, row in df.iterrows():
    if any(word in elect for word in str(row['JIEBA']).split()):
        df.at[index, '电子产品'] = 1  # 将匹配到词典的行设置为电子产品

df['游戏'] = 0  # 创建新列并初始化为0
for index, row in df.iterrows():
    if any(word in game for word in str(row['JIEBA']).split()):
        df.at[index, '游戏'] = 1  # 将匹配到词典的行设置为游戏

df['旅游'] = 0  # 创建新列并初始化为0
for index, row in df.iterrows():
    if any(word in trip for word in str(row['JIEBA']).split()):
        df.at[index, '旅游'] = 1  # 将匹配到词典的行设置为旅游

df['影视'] = 0  # 创建新列并初始化为0
for index, row in df.iterrows():
    if any(word in video for word in str(row['JIEBA']).split()):
        df.at[index, '影视'] = 1  # 将匹配到词典的行设置为影视

# 将修改后的DataFrame保存为新的CSV文件
df.to_csv('modified_csv_file.csv', index=False, encoding='GB18030')  # 保存为新的CSV文件
