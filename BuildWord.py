import self
from gensim.models import word2vec

food = 'similar_words.txt'
pe = 'PE.txt'
hosp = 'hospital.txt'
elect = 'elect.txt'
game = 'game.txt'
trip = 'trip.txt'
video = 'video.txt'


def similar_words(word):
    """计算某个词的相关词列表——前提是已存在训练好的模型"""
    model1 = word2vec.Word2Vec.load("model.bin")  # 模型的加载方式
    y2 = model1.wv.most_similar(word, topn=10)  # 10个最相关的
    result = f""
    # print("和{}最相关的词有：".format(word))
    for item in y2:
        if item[0] == '0':
            break
        # print(item[0], '%.3f' % item[1])
        result += f"{item[0]}\n"  # 只保存词汇，不保存相关性数字
    return result


def delete(path):
    # 读取文件并去除重复词汇
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 去除重复词汇并写入新文件
    unique_lines = list(set(lines))
    with open(path, 'w', encoding='utf-8') as file:
        file.writelines(unique_lines)


'''
# 关键词词库
food_list = ['美食', '小吃', '海鲜', '特色美食', '糕点', '拉面', '特色菜', '一日游', '小吃街', '风情街',
             '特产', '烧烤', '街边', '特色菜', '加盟店', '一条街', '特色小吃', '周边城市', '甜点',
             '西点' '烘焙', '面点', '菜品', '点心', '中餐', '寿司', '烤鱼', '烘培', '牛排', '牛肉面',
             '板鸭', '豆花', '汉堡', '三明治', '猪排', '拌面', '酸辣粉', '年糕', '寿司', '披萨',
             '年糕', '蛋挞', '拉面', '猪排', '鸡胸肉', '梭子蟹', '鸡蛋糕', '薯片', '沙拉酱', '龟苓膏',
             '鸡蛋糕', '咸鸭蛋', '挂面', '八宝粥', '火腿肠', '三明治', '披萨', '蚕豆']
'''


# if __name__ == "__main__":
# similar_words('薯片')
def write_similar(path):
    # 从文件中读取食物列表
    with open(path, 'r', encoding='utf-8') as file:
        food_list = file.read().splitlines()
    with open(path, 'w', encoding='utf-8') as file:
        for food in food_list:
            try:
                # similar_words(food)
                result = similar_words(food)
                file.write(result)
                print(f"食物 '{food}' 的同义词：")
            except KeyError:
                print(f"找不到 '{food}' 的同义词\n")


# write_similar(food)
# delete(food)

write_similar(pe)
delete(pe)
write_similar(elect)
delete(elect)
write_similar(game)
delete(game)
write_similar(hosp)
delete(hosp)
write_similar(trip)
delete(trip)
write_similar(video)
delete(video)