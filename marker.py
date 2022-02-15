# CCPD 数据集转 YOLO 数据集
"""
Train a YOLOv5 model on a custom dataset.

Models and datasets download automatically from the latest YOLOv5 release.
Models: https://github.com/ultralytics/yolov5/tree/master/models
Datasets: https://github.com/ultralytics/yolov5/tree/master/data
Tutorial: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

CCPD Dataset:
https://github.com/detectRecog/CCPD
https://github.com/xiaosongshine/CCPD_Plus
"""

import os
from pathlib import Path

import cv2


def get_bigger_rectangle(fp: Path):
    """
    CCPD 数据集 文件名称使用 '-' 分割, 第三部分 558&578_173&523_159&434_586&474 对应车牌四个顶点坐标(右下角开始顺时针排列)
    右下(558, 578); 左下(173, 523); 左上(159, 434); 右上(586, 474)
    top; bottom; left; right
    :param fp: file path
    :return: 车牌子最靠外的顶点坐标 (min(x), min(y), max(x), max(y))
    """
    border = fp.stem.split("-")[3].split("_")
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


def get_plate_no(fp: Path):
    """获取车牌号码
    CCPD 数据集 文件名称使用 '-' 分割, 第四部分 的每个图像只有一个 LP。每个LP编号由一个汉字、一个字母和五个字母或数字组成。有效的中国车牌由七个字符组成: 省(1个字符)  , 字母 (1个字符), 字母+数字 (5个字符)。
    "0_0_22_27_27_33_16"是每个字符的索引。这三个数组定义如下。每个数组的最后一个字符是字母 O , 而不是数字 0。我们使用O作为"无字符"的标志 , 因为中文车牌字符中没有O。
    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
                "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
    ads       = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
    :param fp: file name
    :return: 车牌号码
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
    b = fp.stem.split("-")[4].split("_")
    plate_no = province[int(b[0])]
    for i in range(1, len(b)):
        plate_no = plate_no + plate_num[int(b[i])]
    return plate_no


def mark_rectangle_number(fp: Path):
    """

    :param fp: file path
    :return:
    """
    # 面积比-斜度-左上右下坐标-四个角坐标 (右下角开始顺时针) -车牌-亮度-模糊度

    img = cv2.imread(str(fp))
    _rect = get_bigger_rectangle(fp)
    plate_no = get_plate_no(fp)
    print(fp.stem, plate_no, _rect)
    # cv2.rectangle(img, pt1, pt2, color, thickness=..., lineType=..., shift=.) 
    cv2.rectangle(img, (_rect[0], _rect[1]), (_rect[2], _rect[3]), (0, 0, 255), 1, )

    add_text = img.copy()
    cv2.putText(add_text, plate_no, (_rect[0], _rect[1]), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2, )

    cv2.namedWindow(plate_no)
    cv2.imshow(plate_no, add_text)
    cv2.waitKey()
    cv2.destroyAllWindows()


def mark2yolo(fp: Path):
    """

    :param fp: directory
    :return:
    """
    img = cv2.imread(str(fp))
    sp = img.shape
    height = sp[0]
    width = sp[1]
    _rect = get_bigger_rectangle(fp)
    lu_1 = _rect[0]
    lu_2 = _rect[1]
    rb_1 = _rect[2]
    rb_2 = _rect[3]

    # 0 中心点 (归一化宽)  中心点 (归一化长)  框宽度 (归一化)  框长度 (归一化) 
    mid_x = (rb_1 + lu_1) / 2 / width
    mid_y = (rb_2 + lu_2) / 2 / height
    dis_x = (rb_1 - lu_1) / width
    dic_y = (rb_2 - lu_2) / height

    res = "0 " + str(mid_x) + " " + str(mid_y) + " " + str(dis_x) + " " + str(dic_y)
    print(res)

    # if not os.path.exists("labels"):
    #     os.makedirs("labels")
    # with open(f"labels/{fp.stem} .txt", "a") as f:
    #     f.write(res)
    #     f.close()


def mark(fp: Path):
    # 参考
    # https://blog.csdn.net/qq_36516958/article/details/114274778
    # https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#2-create-labels
    # _rect = get_bigger_rectangle(fp)
    list1 = fp.stem.split("-", 3)  # 第一次分割 , 以减号'-'做分割
    subname = list1[2]
    lt, rb = subname.split("_", 1)  # 第二次分割 , 以下划线'_'做分割
    lx, ly = lt.split("&", 1)
    rx, ry = rb.split("&", 1)

    width = int(rx) - int(lx)
    height = int(ry) - int(ly)  # bounding box的宽和高
    center_x = float(lx) + width / 2
    center_y = float(ly) + height / 2  # bounding box中心点

    img = cv2.imread(str(fp))
    width = width / img.shape[1]
    height = height / img.shape[0]
    center_x = center_x / img.shape[1]
    center_y = center_y / img.shape[0]
    # 绿牌是第0类 , 蓝牌是第1类
    meta = [str(1), str(center_x), str(center_y), str(width), str(height)]
    # cv2.rectangle(img, pt1, pt2, color, thickness=..., lineType=..., shift=.) 
    # cv2.rectangle(img, (5, 8), (20, 40), (0, 0, 255), 1, )
    # plate_no = get_plate_no(fp)
    # cv2.namedWindow(plate_no)
    # cv2.imshow(plate_no, img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # txtfile = f"{fp.parent}/{fp.stem}.txt"
    # 绿牌是第0类 , 蓝牌是第1类
    # with open(txtfile, "w") as f:
    #     f.write(str(1) + " " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height) + "\n")
    #     f.write(" ".join(meta))

    print(f"mark \tfilename\t:{fp.stem} \timg.shape\t:{img.shape} meta:{meta}")


def mark1(fp: Path):
    """

    :param fp: file Path
    :return:
    """
    # fp = Path(filepath)
    _rect = get_bigger_rectangle(fp)

    # bounding box的宽和高
    width = _rect[2] - _rect[0]
    height = _rect[3] - _rect[1]
    # bounding box中心点
    center_x = _rect[2] / 2
    center_y = _rect[3] / 2

    img = cv2.imread(str(fp))
    width = width / img.shape[1]
    height = height / img.shape[0]
    center_x = center_x / img.shape[1]
    center_y = center_y / img.shape[0]
    # 绿牌是第0类 , 蓝牌是第1类
    meta = [str(1), str(center_x), str(center_y), str(width), str(height)]

    txt_file = f"{fp.parent}/{fp.stem}.txt"
    # 绿牌是第0类 , 蓝牌是第1类
    # with open(txt_file, "w") as f:
    #     f.write(str(1) + " " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height) + "\n")
    #     f.write(" ".join(meta))

    print(f"mark1 \tfilename\t:{fp.stem} \timg.shape\t:{img.shape} meta:{meta}")


if __name__ == "__main__":
    source = "../dataset/CCPD/demo"
    _files = os.listdir(source)
    for fn in _files:
        f = Path(os.path.join(source,fn))
        # mark(f)
        mark1(f)
        mark2yolo(f)
        # mark_rectangle_number(f)
        print(f.stem)
