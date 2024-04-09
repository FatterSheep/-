"""
复赛给的数据十分混乱,csv文件是多行多列的;
这一步是做初步处理,将文件统一格式为多行单列的数据;
方便做后续的处理
"""

import pandas as pd

# 读取CSV文件
train_path = '../data/user_tag_query.10W.csv'
test_path = '../data/user_tag_query.30W.csv'
train_file = '../data/user_tag_query.Train.csv'
test_file = '../data/user_tag_query.Test.csv'


def merge(file_path, output_file):
    # 读取CSV文件
    data = pd.read_csv(file_path, encoding='GB18030', header=None, dtype=str)

    # 将多列数据合并为一列
    # merged_column = pd.concat([data[col] for col in data.columns])  # 这种方式会循环使用已有的列来构建新的行，需要修改

    # 将每列数据添加到一个列表中
    merged_column = []
    for col in data.columns:
        merged_column.extend(data[col])

    # 只保留原始数据的行数
    merged_series = pd.Series(merged_column)[:len(data)]

    # 保存合并后的数据到新的CSV文件
    merged_series.to_csv(output_file, index=False, header=False, encoding='GB18030')
    print('合并完成')


merge(train_path, train_file)
merge(test_path, test_file)

