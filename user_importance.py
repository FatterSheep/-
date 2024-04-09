import pandas as pd

# 读取CSV文件
df = pd.read_csv('modified_csv_file.csv', encoding='GB18030')  # 替换成你的文件路径

# 选取倒数7列并计算它们的总和
last_seven_columns = df.iloc[:, -7:]
sum_last_seven = last_seven_columns.sum(axis=1)

# 在DataFrame中添加新列，包含总和
df['importance'] = sum_last_seven

# 保存DataFrame为CSV文件
df.to_csv('user_importance.csv', index=False, encoding='GB18030')
