# CCPD 数据集转 YOLO 数据集
"""
Train a YOLOv5 model on a custom dataset.

Models and datasets download automatically from the latest YOLOv5 release.
Models: https://github.com/ultralytics/yolov5/tree/master/models
Datasets: https://github.com/ultralytics/yolov5/tree/master/data
Tutorial: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

CCPD Dataset: https://github.com/detectRecog/CCPD
https://blog.csdn.net/ysh1026/article/details/119389985
https://www.codeleading.com/article/79865429542/
https://blog.csdn.net/lswdecsdn/article/details/106676840

"""

import os

import cv2


def get_bigger_rectangle(fn):
    """
    @param fn: file name
    '-' 分割, 第三部分 558&578_173&523_159&434_586&474 对应车牌四个顶点坐标(右下角开始顺时针排列) 
    右下(558, 578); 左下(173, 523); 左上(159, 434); 右上(586, 474)
    top; bottom; left; right
    return: 车牌子最靠外的顶点坐标
    """
    border = fn.split("-")[3].split("_")
    rb_x = int(border[0].split("&")[0])
    rb_y = int(border[0].split("&")[1])

    lb_x = int(border[1].split("&")[0])
    lb_y = int(border[1].split("&")[1])

    lu_x = int(border[2].split("&")[0])
    lu_y = int(border[2].split("&")[1])

    ru_x = int(border[3].split("&")[0])
    ru_y = int(border[3].split("&")[1])

    # 选最靠外的顶点坐标
    return (
        min(rb_x, lb_x, lu_x, ru_x),
        min(rb_y, lb_y, lu_y, ru_y),
        max(rb_x, lb_x, lu_x, ru_x),
        max(rb_y, lb_y, lu_y, ru_y),
    )


def get_plate_no(fn):
    """获取车牌号码
    CCPD 中的每个图像只有一个 LP。每个LP编号由一个汉字、一个字母和五个字母或数字组成。有效的中国车牌由七个字符组成：省（1个字符），字母（1个字符），字母+数字（5个字符）。
    "0_0_22_27_27_33_16"是每个字符的索引。这三个数组定义如下。每个数组的最后一个字符是字母 O，而不是数字 0。我们使用O作为"无字符"的标志，因为中文车牌字符中没有O。
    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
                "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
    ads       = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

    """
    province = {
        0: "皖",
        1: "沪",
        2: "津",
        3: "渝",
        4: "冀",
        5: "晋",
        6: "蒙",
        7: "辽",
        8: "吉",
        9: "黑",
        10: "苏",
        11: "浙",
        12: "京",
        13: "闽",
        14: "赣",
        15: "鲁",
        16: "豫",
        17: "鄂",
        18: "湘",
        19: "粤",
        20: "桂",
        21: "琼",
        22: "川",
        23: "贵",
        24: "云",
        25: "西",
        26: "陕",
        27: "甘",
        28: "青",
        29: "宁",
        30: "新",
        31: "警",
        32: "学",
    }
    plate_num = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "K",
        10: "L",
        11: "M",
        12: "N",
        13: "P",
        14: "Q",
        15: "R",
        16: "S",
        17: "T",
        18: "U",
        19: "V",
        20: "W",
        21: "X",
        22: "Y",
        23: "Z",
        24: "0",
        25: "1",
        26: "2",
        27: "3",
        28: "4",
        29: "5",
        30: "6",
        31: "7",
        32: "8",
        33: "9",
    }
    b = fn.split("-")[4].split("_")
    plate_no = province[int(b[0])]
    for i in range(1, len(b)):
        plate_no = plate_no + plate_num[int(b[i])]
    return plate_no


def rectangle(dir):
    # 面积比-斜度-左上右下坐标-四个角坐标（右下角开始顺时针）-车牌-亮度-模糊度

    files = os.listdir(dir)
    for fn in files:
        img = cv2.imread(os.path.join(dir, fn))

        cv2.rectangle(
            img,
            (get_bigger_rectangle(fn)[0], get_bigger_rectangle(fn)[1]),
            (get_bigger_rectangle(fn)[2], get_bigger_rectangle(fn)[3]),
            (0, 0, 255),
            1,
        )

        add_text = img.copy()
        cv2.putText(
            add_text,
            get_plate_no(fn),
            (get_bigger_rectangle(fn)[0], get_bigger_rectangle(fn)[1]),
            cv2.FONT_HERSHEY_COMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

        cv2.namedWindow("hello, World")
        cv2.imshow("Hello, World", add_text)
        cv2.waitKey()
        cv2.destroyAllWindows()


def main(dir):
    files = os.listdir(dir)
    for n in files:

        img = cv2.imread("../ccpd_base/" + n)
        sp = img.shape
        height = sp[0]
        width = sp[1]

        lu_1 = get_bigger_rectangle(n)[0]
        lu_2 = get_bigger_rectangle(n)[1]
        rb_1 = get_bigger_rectangle(n)[2]
        rb_2 = get_bigger_rectangle(n)[3]

        # 0 中心点（归一化宽） 中心点（归一化长） 框宽度（归一化） 框长度（归一化）
        mid_x = (rb_1 + lu_1) / 2 / width
        mid_y = (rb_2 + lu_2) / 2 / height
        dis_x = (rb_1 - lu_1) / width
        dic_y = (rb_2 - lu_2) / height

        res = "0 " + str(mid_x) + " " + str(mid_y) + " " + str(dis_x) + " " + str(dic_y)
        print(res)

        if not os.path.exists("E:\CCPD\CCPD2019\ccpd_base"):
            os.makedirs("E:\CCPD\CCPD2019\ccpd_base")
        f = open("E:\CCPD\CCPD2019\ccpd_base" + n.split(".")[0] + ".txt", "a")
        f.write(res)
        f.close()


def mark(dir):
    # 参考
    # https://blog.csdn.net/qq_36516958/article/details/114274778
    # https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#2-create-labels

    for filename in os.listdir(dir):
        list1 = filename.split("-", 3)  # 第一次分割，以减号'-'做分割
        subname = list1[2]
        lt, rb = subname.split("_", 1)  # 第二次分割，以下划线'_'做分割
        lx, ly = lt.split("&", 1)
        rx, ry = rb.split("&", 1)
        meta = []

        width = int(rx) - int(lx)
        height = int(ry) - int(ly)  # bounding box的宽和高
        center_x = float(lx) + width / 2
        center_y = float(ly) + height / 2  # bounding box中心点

        img = cv2.imread(os.path.join(dir, filename))
        width = width / img.shape[1]
        height = height / img.shape[0]
        center_x = center_x / img.shape[1]
        center_y = center_y / img.shape[0]
        # 绿牌是第0类，蓝牌是第1类
        meta.append(str(1))
        meta.append(str(center_x))
        meta.append(str(center_y))
        meta.append(str(width))
        meta.append(str(height))

        txtname = filename.split(".", 1)
        txtfile = f"{dir}/{txtname[0]}.txt"
        # 绿牌是第0类，蓝牌是第1类
        with open(txtfile, "w") as f:
            f.write(str(1) + " " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height) + "\n")
            f.write(" ".join(meta))

        print(
            f"\tfilename\t:{filename} \n\ttxtfile\t:{txtfile} \n\timg.shape\t:{img.shape} \n\tcenter_x\t:{center_x} \n\tcenter_y\t:{center_y} \n\twidth\t:{width} \n\theight\t:{height}")


if __name__ == "__main__":
    dir = "../dataset/CCPD2020/ccpd_green/demo"
    # main(dir)
    # rectangle(dir)
    mark(dir)
