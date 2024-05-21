import re

def is_hero_i_format(s):
    # 正则表达式解释：
    # ^hero_          : 字符串必须以"hero_"开始
    #   \d{1,3}       : 紧跟着1到3个数字（因为i的范围是1~100，所以这里限制为3位数字）
    #   (?<=\d)$      : 确保字符串以数字结尾，(?<=\d)是一个正向后查找，确保数字前面没有其他字符
    pattern = r'^hero_\d{1,3}(?<=\d)$'
    return re.match(pattern, s) is not None

# 测试函数
test_strings = ["hero_1", "hero_100", "hero_101", "hero_a", "hero"]
for test_str in test_strings:
    print(f"'{test_str}' is in 'hero_i' format: {is_hero_i_format(test_str)}")