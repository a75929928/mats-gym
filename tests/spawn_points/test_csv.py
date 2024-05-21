import csv

# 打开CSV文件
csv_path = "/home/hjh/carla/mats-gym/spawn_points/Town04.csv"
with open(csv_path, 'r') as csvfile:
    # 创建CSV阅读器
    datareader = csv.reader(csvfile, delimiter=',')

    # 遍历CSV文件中的每一行
    for row in datareader:
        # 假设每行都有三个元素
        index, x, y, z = row
        # 将字符串转换为浮点数
        x = float(x)
        y = float(y)
        z = float(z)
        # 打印出(x, y, z)的形式
        print(f"(x={x}, y={y}, z={z})")
        break