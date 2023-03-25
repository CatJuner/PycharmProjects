import os
from PIL import Image, ImageDraw, ImageFont
import random

# 随机生成4个字符
code = ''.join(random.sample('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', 4))
# 创建一个空白图像
img = Image.new('RGB', (120, 30), (255, 255, 255))
# 获取一个绘图对象
draw = ImageDraw.Draw(img)
# 设置字体
font = ImageFont.FreeTypeFont('KumoFont.ttf', 25)
# 绘制字符
for i in range(4):
    draw.text((30 * i + 10, 0), code[i], (0, 0, 0), font)

# 绘制干扰线
for i in range(5):
    x1 = random.randint(0, 120)
    y1 = random.randint(0, 30)
    x2 = random.randint(0, 120)
    y2 = random.randint(0, 30)
    draw.line((x1, y1, x2, y2), fill=(0, 0, 0))

# 保存图像
img.save('captcha.png')

a = '*'.join([str(i+1) for i in range(10)])
print(eval(a))
input()