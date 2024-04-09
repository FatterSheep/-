import matplotlib.pyplot as plt
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from wordcloud import WordCloud  # 中文分词
import jieba
import imageio
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread
import random
'''
# with open('../1-process_data/jieba/jieba_train_cut.csv', encoding='GB18030', mode='r') as file:
#     myText = file.read()
#     myText = " ".join(jieba.cut(myText))
#     print(myText)
'''
username = 'root'
password = 'Lyy03191298'
hostname = 'localhost'
database_name = 'datamining'

cursorclass = pymysql.cursors.DictCursor

# 建立数据库连接
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='Lyy03191298',
    db='datamining',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)
# 创建 SQLAlchemy 引擎
engine = create_engine(f"mysql+mysqlconnector://{username}:{password}@{hostname}/{database_name}")
# 执行 SQL 查询语句，读取前 500 行数据到 Pandas DataFrame
query = "SELECT * FROM vocabulary LIMIT 50000000"  # 替换为你的表名
#query_1 = "SELECT * FROM vocabulary"
df = pd.read_sql_query(query, engine)
# df_1 =

# 关闭数据库连接
conn.close()
text_data = df['jieba']  # 替换为实际的文本列名称
text = ' '.join(df['jieba']) # 将文本列合并成一个字符串
'''
简单的词云制作
'''
# 制作词云
wordcloud = WordCloud(font_path="msyh.ttc", background_color="white", height=300, width=400).generate(text)

# 图片展示
plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# 将词云图片导出到当前文件夹
wordcloud.to_file("wordCloudMo.png")
print('词云制作完成！')

'''
绘制指定形状的词云
'''

"""
# 导入imageio库中的imread函数，并用这个函数读取本地图片，作为词云形状图片
mk = imageio.imread("chinamap.png")
w = wordcloud.WordCloud(mask=mk)

# 构建并配置词云对象w，注意要加scale参数，提高清晰度
w = wordcloud.WordCloud(width=1000, height=700, background_color='white', font_path='simsun.ttf', mask=mk, scale=15)

# 对来自外部文件的文本进行中文分词，得到string
f = open('新时代中国特色社会主义.txt', encoding='utf-8')
txt = f.read()
txtlist = jieba.lcut(txt)
string = " ".join(txtlist)

# 将string变量传入w的generate()方法，给词云输入文字
wordcloud = w.generate(string)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# 将词云图片导出到当前文件夹
w.to_file('chinamapWordCloud.png')
print('词云制作完成！')
"""