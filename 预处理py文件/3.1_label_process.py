"""
这一步是去除一些属性文件中的异常值;
例如在Id中不能出现中文,
Age,Gender,Education应该都为数字
如果出现非数字/中文,
说明该Data是异常值,应该去除
"""

import csv
import re

Id = '../1-process_data/Train/ID.csv'
Age = '../1-process_data/Train/AGE.csv'
Gender = '../1-process_data/Train/GENDER.csv'
Education = '../1-process_data/Train/EDUCATION.csv'
# QueryList = '../1-process_data/Train/QUERYLIST.csv'


# 读取CSV文件并处理age列
def pro(file, output):
    with open(file, 'r', newline='', encoding='GB18030') as csvfile:
        csv_reader = csv.reader(csvfile)
        processed_rows = []

        for row in csv_reader:
            # 如果ID列中不包含中文字符，则保留该行
            if not re.search('[\u4e00-\u9fff]', row[0]):
                processed_rows.append(row)


    # 将处理后的数据写回CSV文件
    with open(output, 'w', newline='', encoding='GB18030') as output_csvfile:
        csv_writer = csv.writer(output_csvfile)
        csv_writer.writerows(processed_rows)


pro(Id, Id)
pro(Age, Age)
pro(Gender, Gender)
pro(Education, Education)
