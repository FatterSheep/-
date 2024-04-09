import matplotlib.pyplot as plt
import pandas as pd

# 假设有一些数据和对应的标签
Label = '../1-process_data/label/labels_analyse_0.csv'
df = pd.read_csv(Label, encoding='GB18030')


def Age_1():
    # 选取第二列数据
    second_column = df.iloc[:, 1]  # iloc[:, 1] 选择所有行的第二列数据(Age)
    # 统计第二列中等于 0 的值的数量
    count_0 = (second_column == 0).sum()
    print("Number of zeros in the second column:", count_0)
    print(count_0)

    # 统计第二列中等于 1 的值的数量
    count_1 = (second_column == 1).sum()
    print("Number of one in the second column:", count_1)

    # 统计第二列中等于 2 的值的数量
    count_2 = (second_column == 2).sum()
    print("Number of two in the second column:", count_2)

    # 统计第二列中等于 3 的值的数量
    count_3 = (second_column == 3).sum()
    print("Number of three in the second column:", count_3)

    # 统计第二列中等于 4 的值的数量
    count_4 = (second_column == 4).sum()
    print("Number of four in the second column:", count_4)

    # 统计第二列中等于 5 的值的数量
    count_5 = (second_column == 5).sum()
    print("Number of five in the second column:", count_5)

    # 统计第二列中等于 6 的值的数量
    count_6 = (second_column == 6).sum()
    print("Number of six in the second column:", count_6)

    labels = ['Unknown', '0-18', '19-30', '31-40', '41-50', '51-60', '60+']
    values = [count_0, count_1, count_2, count_3, count_4, count_5, count_6]

    # 创建柱状图
    plt.figure(figsize=(10, 6))  # 设置图像尺寸
    plt.bar(labels, values)
    for i, v in enumerate(values):
        plt.annotate(str(v), xy=(i, v), ha='center', va='bottom')
    plt.xlabel('Age')  # 设置 X 轴标签
    plt.ylabel('Count')  # 设置 Y 轴标签
    plt.title('Count of Values in Age')  # 设置标题
    #
    # 保存图像
    plt.savefig('AGE_plot.png')
    # 显示图像
    plt.show()

def Age_2():
    # 选取第二列数据
    second_column = df.iloc[:, 1]  # iloc[:, 1] 选择所有行的第二列数据(Age)

    # 统计第二列中等于 0 的值的数量
    count_0 = (second_column == 0).sum()

    # 统计第二列中等于 1 的值的数量
    count_1 = (second_column == 1).sum()

    # 统计第二列中等于 2 的值的数量
    count_2 = (second_column == 2).sum()

    # 统计第二列中等于 3 的值的数量
    count_3 = (second_column == 3).sum()

    # 统计第二列中等于 4 的值的数量
    count_4 = (second_column == 4).sum()

    # 统计第二列中等于 5 的值的数量
    count_5 = (second_column == 5).sum()

    # 统计第二列中等于 6 的值的数量
    count_6 = (second_column == 6).sum()

    labels_2 = ['0-18', '19-30', '31-40', '41-50', '51-60', '60+']
    values_2 = [count_1, count_2, count_3, count_4, count_5, count_6]

    # 创建柱状图
    plt.figure(figsize=(10, 6))  # 设置图像尺寸
    plt.bar(labels_2, values_2)
    for i, v in enumerate(values_2):
        plt.annotate(str(v), xy=(i, v), ha='center', va='bottom')
    plt.xlabel('Age')  # 设置 X 轴标签
    plt.ylabel('Count')  # 设置 Y 轴标签
    plt.title('Count of Values in Age')  # 设置标题
    #
    # 保存图像
    plt.savefig('AGE_plot_2.png')
    # 显示图像
    plt.show()

def Gender_1():
    # 选取第三列数据
    third_column = df.iloc[:, 2]  # iloc[:, 2] 选择所有行的第三列数据(GENDER)
    # 统计第二列中等于 0 的值的数量
    Gcount_0 = (third_column == 0).sum()

    # 统计第二列中等于 1 的值的数量
    Gcount_1 = (third_column == 1).sum()

    # 统计第二列中等于 2 的值的数量
    Gcount_2 = (third_column == 2).sum()

    Glabels = ['Unknown', 'Male', 'Female']
    Gvalues = [Gcount_0, Gcount_1, Gcount_2]

    # 创建柱状图
    plt.figure(figsize=(10, 6))  # 设置图像尺寸
    plt.bar(Glabels, Gvalues)
    for i, v in enumerate(Gvalues):
        plt.annotate(str(v), xy=(i, v), ha='center', va='bottom')
    plt.xlabel('Gender')  # 设置 X 轴标签
    plt.ylabel('Count')  # 设置 Y 轴标签
    plt.title('Count of Values in Gender')  # 设置标题
    # 保存图像
    plt.savefig('Gender_plot.png')
    # 显示图像
    plt.show()

def Gender_2():
    # 选取第三列数据
    third_column = df.iloc[:, 2]  # iloc[:, 2] 选择所有行的第三列数据(GENDER)
    # 统计第二列中等于 0 的值的数量
    Gcount_0 = (third_column == 0).sum()

    # 统计第二列中等于 1 的值的数量
    Gcount_1 = (third_column == 1).sum()

    # 统计第二列中等于 2 的值的数量
    Gcount_2 = (third_column == 2).sum()

    Glabels_2 = ['Male', 'Female']
    Gvalues_2 = [Gcount_1, Gcount_2]
    # 创建柱状图
    plt.figure(figsize=(10, 6))  # 设置图像尺寸
    plt.bar(Glabels_2, Gvalues_2)
    for i, v in enumerate(Gvalues_2):
        plt.annotate(str(v), xy=(i, v), ha='center', va='bottom')
    plt.xlabel('Gender')  # 设置 X 轴标签
    plt.ylabel('Count')  # 设置 Y 轴标签
    plt.title('Count of Values in Gender')  # 设置标题
    # 保存图像
    plt.savefig('Gender_plot_2.png')
    # 显示图像
    plt.show()

def Education_1():
    # 选取第四列数据
    forth_column = df.iloc[:, 3]  # iloc[:, 1] 选择所有行的第四列数据(Education)

    # 统计第二列中等于 0 的值的数量
    Ecount_0 = (forth_column == 0).sum()

    # 统计第二列中等于 1 的值的数量
    Ecount_1 = (forth_column == 1).sum()

    # 统计第二列中等于 2 的值的数量
    Ecount_2 = (forth_column == 2).sum()

    # 统计第二列中等于 3 的值的数量
    Ecount_3 = (forth_column == 3).sum()

    # 统计第二列中等于 4 的值的数量
    Ecount_4 = (forth_column == 4).sum()

    # 统计第二列中等于 5 的值的数量
    Ecount_5 = (forth_column == 5).sum()

    # 统计第二列中等于 6 的值的数量
    Ecount_6 = (forth_column == 6).sum()

    Elabels = ['Unknown', 'Primary school', ' junior high school', 'high school',
               'university', 'master', 'doctoral']
    Evalues = [Ecount_0, Ecount_1, Ecount_2, Ecount_3, Ecount_4, Ecount_5, Ecount_6]

    # 创建柱状图
    plt.figure(figsize=(10, 6))  # 设置图像尺寸
    plt.bar(Elabels, Evalues)
    for i, v in enumerate(Evalues):
        plt.annotate(str(v), xy=(i, v), ha='center', va='bottom')
    plt.xlabel('Education')  # 设置 X 轴标签
    plt.ylabel('Count')  # 设置 Y 轴标签
    plt.title('Count of Values in Education')  # 设置标题
    # 保存图像
    plt.savefig('Education_plot.png')
    # 显示图像
    plt.show()

def Education_2():
    # 选取第四列数据
    forth_column = df.iloc[:, 3]  # iloc[:, 3] 选择所有行的第四列数据(Education)

    # 统计第二列中等于 0 的值的数量
    Ecount_0 = (forth_column == 0).sum()

    # 统计第二列中等于 1 的值的数量
    Ecount_1 = (forth_column == 1).sum()

    # 统计第二列中等于 2 的值的数量
    Ecount_2 = (forth_column == 2).sum()

    # 统计第二列中等于 3 的值的数量
    Ecount_3 = (forth_column == 3).sum()

    # 统计第二列中等于 4 的值的数量
    Ecount_4 = (forth_column == 4).sum()

    # 统计第二列中等于 5 的值的数量
    Ecount_5 = (forth_column == 5).sum()

    # 统计第二列中等于 6 的值的数量
    Ecount_6 = (forth_column == 6).sum()
    Elabels_2 = ['Primary school', ' junior high school', 'high school',
                 'university', 'master', 'doctoral']
    Evalues_2 = [Ecount_1, Ecount_2, Ecount_3, Ecount_4, Ecount_5, Ecount_6]

    # 创建柱状图
    plt.figure(figsize=(10, 6))  # 设置图像尺寸
    plt.bar(Elabels_2, Evalues_2)
    for i, v in enumerate(Evalues_2):
        plt.annotate(str(v), xy=(i, v), ha='center', va='bottom')
    plt.xlabel('Education')  # 设置 X 轴标签
    plt.ylabel('Count')  # 设置 Y 轴标签
    plt.title('Count of Values in Education')  # 设置标题
    # 保存图像
    plt.savefig('Education_plot_2.png')
    # 显示图像
    plt.show()

def Age_Education():
    second_column = df.iloc[:, 1]  # iloc[:, 1] 选择所有行的第二列数据(Age)
    forth_column = df.iloc[:, 3]  # iloc[:, 3] 选择所有行的第四列数据(Education)
    data = {
        'Age':second_column,
        'Education':forth_column
    }
    df2 = pd.DataFrame(data)
    # 统计各个受教育程度下的年龄
    education_age = df2.groupby('Education')['Age'].value_counts().unstack().fillna(0)

    # 获取年龄段的数量
    num_age_groups = len(education_age.columns)

    # 选择颜色映射
    colors = plt.cm.get_cmap('tab20', num_age_groups)

    # 绘制柱状图
    plt.figure(figsize=(8, 6))
    education_age.plot(kind='bar', color='skyblue')

    # # 在柱状图上标出数值
    # for i, age_count in enumerate(education_age.values):
    #     for j, count in enumerate(age_count):
    #         plt.text(i, sum(age_count[:j]) + count / 2, str(int(count)), ha='center', va='bottom')

    # 绘制柱状图
    # plt.figure(figsize=(12, 8))
    # for i, age_group in enumerate(education_age.columns):
    #     plt.bar(range(len(education_age)), education_age[age_group],
    #             label=age_group, color=colors(i), alpha=0.8)

    # for i, age_group in enumerate(education_age.columns):
    #     plt.bar(range(len(education_age)), education_age[age_group],
    #             label=age_group, color=colors(i), alpha=0.8)
    # 设置图表标签
    plt.title('Count Age by Education Level')
    plt.xlabel('Education Level')
    plt.ylabel('Count Age')
    plt.tight_layout()
    plt.savefig('Education_Age.png')
    plt.show()
Age_Education()