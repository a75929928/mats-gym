import random

def select_unique_pairs(n, k):
    # 从1到n的整数列表
    x_values = list(range(1, n + 1))
    # 随机选择k个不重复的x值
    chosen_x = random.sample(x_values, k)
    # 初始化y值的列表
    chosen_y = []
    # 用于确保y值不重复
    y_values = list(range(1, n + 1))

    # 为每个x值随机选择一个不重复的y值
    for x in chosen_x:
        y = random.choice(y_values)
        chosen_y.append(y)
        y_values.remove(y)  # 确保y值不重复

    # 组合成对并返回结果
    pairs = list(zip(chosen_x, chosen_y))
    return pairs

# 示例
# n = 10
# k = 3
# print(select_unique_pairs(n, k))
a = [1, 2, 3]
print(a[0])