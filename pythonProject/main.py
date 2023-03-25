import math
import os
from collections import defaultdict

import cv2
from PIL import Image


def del_threshold_noise(image, threshold):
    """
    直接通过阈值去除浅色点(作为第一步)，也要先转化为灰度
    """
    pixel_matrix = image.load()  # load之后可以直接操作，相当于转化为数组了
    rows, cols = image.size
    for col in range(0, cols):
        for row in range(0, rows):
            if pixel_matrix[row, col] >= threshold:
                pixel_matrix[row, col] = 255
    return image


def get_threshold(image):
    """
    获取像素点最多的像素(先转化为灰度)，就是二值化
    """
    pixel_dict = defaultdict(int)
    rows, cols = image.size
    for i in range(rows):
        for j in range(cols):
            pixel = image.getpixel((i, j))
            pixel_dict[pixel] += 1

    count_max = max(pixel_dict.values())
    pixel_dict_reverse = {v: k for k, v in pixel_dict.items()}
    threshold = pixel_dict_reverse[count_max]
    return threshold


def get_bin_table(threshold):
    """
    按照阈值进行二值化(先转化为灰度后，再进行二值化)
    :param threshold: 像素阈值
    :return:
    """
    table = []
    for i in range(256):  # 0~256
        rate = 0.1
        if threshold * (1 - rate) <= i <= threshold * (1 + rate):
            table.append(1)  # 白色
        else:
            table.append(0)  # 黑色

    # for i in range(256):  # 或者不做判断，直接找个界限（只找出黑色内容即可）
    #     if i < threshold:
    #         table.append(0)
    #     else:
    #         table.append(1)
    return table


def del_cut_noise(im_cut):
    '''
    通过颜色区分：将图切为小图，找第二多颜色的像素，从而去除干扰线（先转化为灰度，入参和出参都是ndarray格式）
    variable：bins：灰度直方图bin的数目
              num_gray:像素间隔
    method：1.找到灰度直方图中像素第二多所对应的像素
            2.计算mode
            3.除了在mode+-一定范围内的，全部变为空白。
    '''
    bins = 16  # 直方图柱子的数量，每个柱子都有一定的范围
    num_gray = math.ceil(256 / bins)  # 像素间隔就是柱子的范围
    hist = cv2.calcHist([im_cut], [0], None, [bins], [0, 256])
    lists = []
    for i in range(len(hist)):
        # print hist[i][0]
        lists.append(hist[i][0])
    second_max = sorted(lists)[-2]  # 第二多的像素
    bins_second_max = lists.index(second_max)  # 第二多的像素是第几个柱子

    mode = (bins_second_max + 0.5) * num_gray  # 取柱子的中间（平均），比如2.5， 总的结果就是：第二多的平均的像素

    for i in range(len(im_cut)):
        for j in range(len(im_cut[0])):
            if im_cut[i][j] < mode - 20 or im_cut[i][j] > mode + 20:  # 数组可以直接操作
                # print im_cut[i][j]
                im_cut[i][j] = 255
    return im_cut


def del_dot_noise(image):
    """
    干扰点降噪
    :param image:
    :return:
    """
    rows, cols = image.size  # 图片的宽度和高度
    change_pos = []  # 记录噪声点位置
    # 遍历图片中的每个点，除掉边缘
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # pixel_set用来记录该店附近的黑色像素的数量
            pixel_set = []
            # 取该点的邻域为以该点为中心的九宫格
            for m in range(i - 1, i + 2):
                for n in range(j - 1, j + 2):
                    if image.getpixel((m, n)) != 1:  # 1为白色,0位黑色
                        pixel_set.append(image.getpixel((m, n)))

            # 如果该位置的九宫内的黑色数量小于等于4，则判断为噪声
            if len(pixel_set) <= 4:
                change_pos.append((i, j))

    # 对相应位置进行像素修改，将噪声处的像素置为1（白色）
    for pos in change_pos:
        image.putpixel(pos, 1)  # 找到之后一起删除，而不是一个个删除

    return image


def remove_noise_line(image):
    """
    去除验证码干扰线（操作像素点：随机应变）
    :param image:
    :return:
    """
    global min, max
    try:
        width, height = image.size
        total_list = []
        for i in range(width):
            dot_list = []
            noise_dot_list = []
            for j in range(height):
                if image.getpixel((i, j)) < 200:
                    dot_list.append((i, j))

            if i == 0:
                if len(dot_list) == 1:
                    total_list.append(dot_list[0])
                    max = dot_list[0][1]
                    min = dot_list[0][1]

                elif len(dot_list) == 2:
                    if dot_list[1][1] == dot_list[0][1] + 1:
                        total_list.append(dot_list[0])
                        total_list.append(dot_list[1])
                        max = dot_list[1][1]
                        min = dot_list[0][1]
                    elif abs(dot_list[0][1] - dot_list[1][1]) == 2:
                        total_list.append(dot_list[0])
                        total_list.append(dot_list[1])
                        max = dot_list[1][1]
                        min = dot_list[0][1]

                elif len(dot_list) == 3:
                    if dot_list[1][1] == dot_list[0][1] + 1 and dot_list[2][1] == dot_list[0][1] + 2:
                        total_list.append(dot_list[0])
                        total_list.append(dot_list[1])
                        total_list.append(dot_list[2])
                        max = dot_list[2][1]
                        min = dot_list[0][1]

            for m in dot_list:
                if m[1] in range(min - 1, max + 2):
                    noise_dot_list.append(m)
                # if max + 2 - min > 8:
                #     if m[1] in range(total_list[-2][1]-5, total_list[-2][1]+6):
                #         noise_dot_list.append(m)
                # else:
                #     if m[1] in range(min-1, max+2):
                #         noise_dot_list.append(m)

            noise_dot_list1 = []
            noise_dot_list2 = []
            # print('noise_dot_list', noise_dot_list)
            if noise_dot_list:
                if len(noise_dot_list) == noise_dot_list[-1][1] - noise_dot_list[0][1] + 1:
                    pass
                else:
                    for index, value in enumerate(noise_dot_list):
                        if index > len(noise_dot_list) - 2:
                            break

                        if index == 0:
                            if value[1] + 1 == noise_dot_list[index + 1][1] and value[1] + 2 != \
                                    noise_dot_list[index + 2][1]:
                                noise_dot_list1.append(value) if value not in noise_dot_list1 else 1
                                noise_dot_list1.append(noise_dot_list[index + 1]) if noise_dot_list[index + 1] not in noise_dot_list1 else 1

                        elif index == len(noise_dot_list) - 2:
                            if value[1] + 1 == noise_dot_list[index + 1][1]:
                                noise_dot_list1.append(value) if value not in noise_dot_list1 else 1
                                noise_dot_list1.append(noise_dot_list[index + 1]) if noise_dot_list[index + 1] not in noise_dot_list1 else 1
                        else:
                            if value[1] + 1 == noise_dot_list[index + 1][1] and value[1] + 2 != \
                                    noise_dot_list[index + 2][1] and value[1] - 1 != noise_dot_list[index - 1][1]:
                                noise_dot_list1.append(value) if value not in noise_dot_list1 else 1
                                noise_dot_list1.append(noise_dot_list[index + 1]) if noise_dot_list[index + 1] not in noise_dot_list1 else 1

                    for index, value in enumerate(noise_dot_list):
                        if index > len(noise_dot_list) - 3:
                            break

                        if index == 0:
                            if value[1] + 1 == noise_dot_list[index + 1][1] and value[1] + 2 == \
                                    noise_dot_list[index + 2][1] and value[1] + 3 != noise_dot_list[index + 3][1]:
                                noise_dot_list1.append(value) if value not in noise_dot_list1 else 1
                                noise_dot_list1.append(noise_dot_list[index + 1]) if noise_dot_list[index + 1] not in noise_dot_list1 else 1
                                noise_dot_list1.append(noise_dot_list[index + 2]) if noise_dot_list[index + 2] not in noise_dot_list1 else 1


                        elif index == len(noise_dot_list) - 3:
                            if value[1] + 1 == noise_dot_list[index + 1][1] and value[1] + 2 == \
                                    noise_dot_list[index + 2][1] and value[1] - 1 != noise_dot_list[index - 1][1]:
                                noise_dot_list1.append(value) if value not in noise_dot_list1 else 1
                                noise_dot_list1.append(noise_dot_list[index + 1]) if noise_dot_list[index + 1] not in noise_dot_list1 else 1
                                noise_dot_list1.append(noise_dot_list[index + 2]) if noise_dot_list[index + 2] not in noise_dot_list1 else 1

                        else:
                            if value[1] + 1 == noise_dot_list[index + 1][1] and value[1] + 2 == \
                                    noise_dot_list[index + 2][1] and value[1] + 3 != noise_dot_list[
                                index + 3][1] and value[1] - 1 != noise_dot_list[index - 1][1]:
                                noise_dot_list1.append(value) if value not in noise_dot_list1 else 1
                                noise_dot_list1.append(noise_dot_list[index + 1]) if noise_dot_list[index + 1] not in noise_dot_list1 else 1
                                noise_dot_list1.append(noise_dot_list[index + 2]) if noise_dot_list[index + 2] not in noise_dot_list1 else 1

                # 找最近的两个或者三个
                # print('total_list', total_list)
                if noise_dot_list1:
                    d_value = sorted([abs(total_list[-2][1] - l[1]) for l in noise_dot_list1])[0]
                    mark = sorted([(total_list[-2][1] - l[1]) for l in noise_dot_list1])
                    if d_value in mark:
                        # print(total_list[-2][1] - d_value - 2)
                        # print(total_list[-2][1] -d_value + 3)
                        for i in noise_dot_list1:
                            if i[1] in range(total_list[-2][1] - d_value - 2, total_list[-2][1] - d_value + 3):
                                noise_dot_list2.append(i)
                    else:
                        # print(total_list[-2][1] + d_value - 2)
                        # print(d_value + total_list[-2][1] + 3)
                        for i in noise_dot_list1:
                            if i[1] in range(total_list[-2][1] + d_value - 2, d_value + total_list[-2][1] + 3):
                                noise_dot_list2.append(i)

                # print('noise_dot_list1', noise_dot_list1)
                # print('noise_dot_list2', noise_dot_list2)

                if not noise_dot_list2:
                    count = 0
                    if noise_dot_list[0][1] != 0 and noise_dot_list[-1][1] != 39:

                        if image.getpixel((noise_dot_list[0][0], noise_dot_list[0][1] - 1)) < 200:
                            count += 1

                        if image.getpixel((noise_dot_list[-1][0], noise_dot_list[-1][1] + 1)) < 200:
                            count += 1

                        if len(noise_dot_list) + count < 4:
                            for n in noise_dot_list:
                                total_list.append(n)
                else:
                    for n in noise_dot_list2:
                        total_list.append(n)

                min = noise_dot_list[0][1]
                max = noise_dot_list[-1][1]
            else:
                pass

        # print(total_list)
        for pos in total_list:
            image.putpixel(pos, 255)
    except Exception as e:
        print(e)

    return image


# 图像转为二进制
def binarizing_point(img, threshold):  # 遍历像素点，以一定阈值为界限，把图片变成二值图像，要么纯黑0，要么纯白255
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    img = img.point(table, '1')
    return img


def binarizing_load(img, threshold):
    pixdata = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img


def del_other_dots(img):
    pixdata = img.load()
    w, h = img.size
    for i in range(h):  # 最左列和最右列
        # print(pixdata[0, i]) # 最左边一列的像素点信息
        # print(pixdata[w-1, i]) # 最右边一列的像素点信息
        if pixdata[0, i] == 0 and pixdata[1, i] == 255:
            pixdata[0, i] = 255
        if pixdata[w - 1, i] == 0 and pixdata[w - 2, i] == 255:
            pixdata[w - 1, i] = 255

    for i in range(w):  # 最上行和最下行
        # print(pixdata[i, 0]) # 最上边一行的像素点信息
        # print(pixdata[i, h-1]) # 最下边一行的像素点信息
        if pixdata[i, 0] == 0 and pixdata[i, 1] == 255:
            pixdata[i, 0] = 255
        if pixdata[i, h - 1] == 0 and pixdata[i, h - 2] == 255:
            pixdata[i, h - 1] = 255

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if pixdata[x, y] == 0:  # 遍历除了四个边界之外的像素黑点
                count = 0  # 统计某个黑色像素点周围九宫格中白块的数量（最多8个）
                if pixdata[x + 1, y + 1] == 255:
                    count = count + 1
                if pixdata[x + 1, y] == 255:
                    count = count + 1
                if pixdata[x + 1, y - 1] == 255:
                    count = count + 1
                if pixdata[x, y + 1] == 255:
                    count = count + 1
                if pixdata[x, y - 1] == 255:
                    count = count + 1
                if pixdata[x - 1, y + 1] == 255:
                    count = count + 1
                if pixdata[x - 1, y] == 255:
                    count = count + 1
                if pixdata[x - 1, y - 1] == 255:
                    count = count + 1

                if count > 3:
                    print('位置：(' + str(x) + ', ' + str(y) + ')----' + str(count))
                    pixdata[x, y] = 255

    for i in range(h):  # 最左列和最右列
        if pixdata[0, i] == 0 and pixdata[1, i] == 255:
            pixdata[0, i] = 255
        if pixdata[w - 1, i] == 0 and pixdata[w - 2, i] == 255:
            pixdata[w - 1, i] = 255

    for i in range(w):  # 最上行和最下行
        if pixdata[i, 0] == 0 and pixdata[i, 1] == 255:
            pixdata[i, 0] = 255
        if pixdata[i, h - 1] == 0 and pixdata[i, h - 2] == 255:
            pixdata[i, h - 1] = 255
    return img


if __name__ == '__main__':
    # 顺序：先灰度 --- 二值化 --- 降噪
    # （如果通过点或者线降噪，降噪可以在二值化后，如果通过颜色降噪，要在二值化之前）
    for i in os.listdir('C:/Users/Catjuner/PycharmProjects/pythonProject/test_images/'):
        # print(i)
        im = Image.open('C:/Users/Catjuner/PycharmProjects/pythonProject/test_images/' + i)
        # im = Image.open(io.BytesIO(f)) # f位读取的二进制
        im = im.convert('L')

        im = del_threshold_noise(im, 120)

        # table = get_bin_table(240)
        # out = im.point(table, '1')
        # out = del_dot_noise(out)
        out = remove_noise_line(im)

        out.save('C:/Users/Catjuner/PycharmProjects/pythonProject/images_store/' + i)

    for i in range(50):
        for i in os.listdir('C:/Users/Catjuner/PycharmProjects/pythonProject/test_images/'):
            img = Image.open('C:/Users/Catjuner/PycharmProjects/pythonProject/test_images/' + i).convert('L')
            image = binarizing_point(img, 160)
            ima = del_other_dots(image)
            # img.save("C://Users//Catjuner//PycharmProjects//pythonProject//" + 'temp' + str(i + 1) + "_p.png")
            # image.save("C://Users//Catjuner//PycharmProjects//pythonProject//" + 'temp' + str(i + 1) + "_b.png")
            ima.save("C:/Users/Catjuner/PycharmProjects/pythonProject/comparision/" + i)

# img = Image.open("C://Users//sckj//Pictures//logo.jpg").convert('L')
# # img.save("C://Users//sckj//Pictures//logoL.jpg")
# image_p = binarizing_point(img, 200)
# image_p.save("C://Users//sckj//Pictures//logoB_p.jpg")

# 验证码数字模板 0-1 矩阵的创建
#
#     num_info = [([0] * 26) for i in range(
#         38)]  # 创建一个宽度为26，高度为38的二维数组，参考：https://www.cnblogs.com/btchenguang/archive/2012/01/30/2332479.html
#
#     pixdata = img.load()
#     for y in range(38):
#         for x in range(26):
#             if pixdata[x, y] == 0:
#                 # print(x, y)
#                 num_info[y][x] = 1  # 注意二维数组中坐标是相反的
#     num_info_list.append(num_info)
#
# # for i in range(10):
# # 	print(num_info_list[i])
#
# f = open('1.txt', 'w')
# f.write(str(num_info_list))
# f.close()
