import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('../seperate/labels_all.csv', encoding='GB18030')  # 请替换为您的文件路径和合适的编码方式

# 检查 JIEBA_Length 列中小于 100 的值并删除对应的行
df = df[df['JIEBA_Length'] >= 100]

# 如果需要将处理后的 DataFrame 保存回 CSV 文件中，可以使用如下方法
df.to_csv('../seperate/labels_all_process.csv', index=False, encoding='GB18030')  # 将处理后的数据保存到新的 CSV 文件中
