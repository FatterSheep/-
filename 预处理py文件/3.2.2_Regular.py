# 对每个单元格去除所有的数字和英文字符
import re

import pandas as pd

file = '../Train/Process_Query.csv'


def remove_alphanumeric(text):
    return re.sub(r'[a-zA-Z0-9]', '', text)


# 读取CSV文件并过滤空白行
data = pd.read_csv(file, encoding='GB18030')
data.dropna(how='all', inplace=True)
# 应用函数到每个单元格
data = data.map(remove_alphanumeric)
# 保存处理后的数据到新的CSV文件
data.to_csv(file, index=False, encoding='GB18030')
