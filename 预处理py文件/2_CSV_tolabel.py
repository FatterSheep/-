# coding=utf-8
"""
根据上一步骤得到的CSV文件，将搜索文本以及三个属性剥离，保存为相应的文件
这些属性文件后续能用得上;
注意路径
"""
import csv

import pandas as pd

# path of the train and test files
# testname1 = '../data/test.csv'
trainname = '../1-process_data/data/user_tag_query.Train.csv'
id = []
age = []
gender = []
education = []
querylist = []

with open(trainname, 'r', newline='', encoding='GB18030') as csvfile:
    # reader = csv.reader(csvfile, delimiter='\t')  # 使用逗号作为分隔符
    target_rows = 100000  # 指定读取的行数
    current_row = 0
    for line in csvfile:
        if current_row < target_rows:
            #print(row_data)
            row_data = line.strip().split('\t')
            if len(row_data) >= 4:  # 检查行中至少有一个元素
                ID = row_data[0]
                id.append(ID)
                print(current_row)
                Age = row_data[1]
                age.append(Age)
                Gender = row_data[2]
                gender.append(Gender)
                print(Gender)
                Education = row_data[3]
                education.append(Education)
                print(Education)
                Rest_of_data = row_data[4:]  # 从第五列开始获取数据
                querylist.append(Rest_of_data)
                print(Rest_of_data)
                current_row += 1
            else:
                print("error")
        else:
            break  # 达到指定行数后停止读取

with open('../Train/ID.csv', 'w', newline='', encoding='GB18030') as file:
    writer = csv.writer(file)
    writer.writerow(['ID'])  # 标题行为ID
    for item in id:
        writer.writerow([item])  # 将每个元素单独写入一行

with open('../Train/AGE.csv', 'w', newline='', encoding='GB18030') as file:
    writer = csv.writer(file)
    writer.writerow(['AGE'])  # 标题行为Age
    for item in age:
        writer.writerow([item])  # 将每个元素单独写入一行
with open('../Train/GENDER.csv', 'w', newline='', encoding='GB18030') as file:
    writer = csv.writer(file)
    writer.writerow(['GENDER'])  # 标题行为GENDER
    for item in gender:
        writer.writerow([item])  # 将每个元素单独写入一行
with open('../Train/EDUCATION.csv', 'w', newline='', encoding='GB18030') as file:
    writer = csv.writer(file)
    writer.writerow(['EDUCATION'])  # 标题行为EDUCATION
    for item in education:
        writer.writerow([item])  # 将每个元素单独写入一行
# 把搜索词拎出来
with open('../Train/QUERYLIST.csv', 'w', newline='', encoding='GB18030') as file:
    writer = csv.writer(file)
    writer.writerow(['QUERYLIST'])  # 标题行为QUERYLIST
    for item in querylist:
        writer.writerow([item])  # 将每个元素单独写入一行

