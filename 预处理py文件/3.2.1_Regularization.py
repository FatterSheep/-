"""
为后续分词进行初步处理,
上一步保存进来的QueryList文件是List形式,
所以保存进来的数据会有[],
这样的符号在jieba分词中是没有意义的,
所以在这一步删除掉
"""
import csv
import re
import string

import pandas as pd

# 读取CSV文件
input_file = '../Train/QUERYLIST.csv'
output_file = '../Train/Process_Query.csv'





with open(input_file, 'r', newline='', encoding='GB18030') as file:
    reader = csv.reader(file)
    # data = [row for row in reader]
    data = [row for row in reader if any(cell.strip() for cell in row)]

# 删除所有的方括号 "[]"
# processed_data = [[cell.replace('[', '').replace(']', '') for cell in row] for row in data]
'''
正则化
'''
# 删除所有的方括号, 单引号、双引号、逗号、句号、和分号
processed_data = [
    [
        cell.replace('[', '')
        .replace(']', '')
        .translate(str.maketrans('', '', string.punctuation))
        .translate(str.maketrans('', '', string.punctuation))
        for cell in row
    ]
    for row in data
]

# clean_text = re.sub(r'[a-zA-Z0-9]', '', processed_data)
# 将处理后的数据保存为CSV文件
with open(output_file, 'w', newline='', encoding='GB18030') as file:
    writer = csv.writer(file)
    writer.writerows(processed_data)

# 读取CSV文件并过滤空白行
data = pd.read_csv(output_file, encoding='GB18030')
data.dropna(how='all', inplace=True)

# 保存处理后的数据到新的CSV文件
data.to_csv(output_file, index=False, encoding='GB18030')
