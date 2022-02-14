# ccpd数据集
CCPD是一个大型的、多样化的、经过仔细标注的中国城市车牌开源数据集。CCPD数据集主要分为CCPD2019数据集和CCPD2020(CCPD-Green)数据集。CCPD2019数据集车牌类型仅有普通车牌(蓝色车牌)，CCPD2020数据集车牌类型仅有新能源车牌(绿色车牌)。
在CCPD数据集中，每张图片仅包含一张车牌，车牌的车牌省份主要为皖。CCPD中的每幅图像都包含大量的标注信息，但是CCPD数据集没有专门的标注文件，每张图像的文件名就是该图像对应的数据标注。标注最困难的部分是注释四个顶点的位置。为了完成这项任务，数据发布者首先在10k图像上手动标记四个顶点的位置。然后设计了一个基于深度学习的检测模型，在对该网络进行良好训练后，对每幅图像的四个顶点位置进行自动标注。最后，数据发布者雇用了7名兼职工人在两周内纠正这些标注。CCPD提供了超过250k个独特的车牌图像和详细的注释。每张图像的分辨率为720(宽度)× 1160(高)× 3(通道)。实际上，这种分辨率足以保证每张图像中的车牌清晰可辨,但是该数据有些图片标注可能不准。不过总的来说CCPD数据集非常推荐研究车牌识别算法的人员学习使用。

CCPD官方开源仓库地址为[CCPD](https://github.com/detectRecog/CCPD)，该仓库介绍了CCPD2019和CCPD2020的相关信息和下载地址。关于CCPD数据集更详细的介绍见其ECCV2018发表论文，地址为Towards [End-to-End License Plate Detection and Recognition: A Large Dataset](https://openaccess.thecvf.com/content_ECCV_2018/papers/Zhenbo_Xu_Towards_End-to-End_License_ECCV_2018_paper.pdf)。


## CCPD数据集介绍
### CCPD2019数据集
CCPD2019数据集主要采集于合肥市停车场，采集时间为上午7:30到晚上10:00，停车场采集人员手持Android POS机对停车场的车辆拍照进行数据采集。所拍摄的车牌照片涉及多种复杂环境，包括模糊、倾斜、雨天、雪天等。CCPD2019数据集包含了25万多幅中国城市车牌图像和车牌检测与识别信息的标注。主要介绍如下：


| 类别           | 描述                                    | 图片数 |
| -------------- | --------------------------------------- | ------ |
| CCPD-Base      | 通用车牌图片                            | 200k   |
| CCPD-FN        | 车牌离摄像头拍摄位置相对较近或较远      | 20k    |
| CCPD-DB        | 车牌区域亮度较亮、较暗或者不均匀        | 20k    |
| CCPD-Rotate    | 车牌水平倾斜20到50度，竖直倾斜-10到10度 | 10k    |
| CCPD-Tilt      | 车牌水平倾斜15到45度，竖直倾斜15到45度  | 10k    |
| CCPD-Weather   | 车牌在雨雪雾天气拍摄得到                | 10k    |
| CCPD-Challenge | 在车牌检测识别任务中较有挑战性的图片    | 10k    |
| CCPD-Blur      | 由于摄像机镜头抖动导致的模糊车牌图片    | 5k     |
| CCPD-NP        | 没有安装车牌的新车图片                  | 5k     |

CCPD2019/CCPD-Base中的图像被拆分为train/val数据集。使用CCPD2019中的子数据集(CCPD-DB、CCPD-Blur、CCPD-FN、CCPD-Rotate、CCPD-Tilt、CCPD-Challenge)进行测试。CCPD2019数据集(数据大小12.26G)
下载地址：
- [谷歌云盘](https://drive.google.com/open?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc)
- [百度云盘](https://pan.baidu.com/share/init?surl=i5AOjAbtkwb17Zy-NQGqkw) 代码：hm0u


## CCPD2020数据集
CCPD2020数据集采集方法应该和CCPD2019数据集类似。CCPD2020仅仅有新能源车牌图片，包含不同亮度，不同倾斜角度，不同天气环境下的车牌。CCPD2020中的图像被拆分为train/val/test数据集，train/val/test数据集中图片数分别为5769/1001/5006。CCPD2020数据集(数据大小865.7MB)下载地址：

- [谷歌云盘](https://drive.google.com/file/d/1m8w1kFxnCEiqz_-t2vTcgrgqNIv986PR/view?usp=sharing)
- [百度云盘](https://pan.baidu.com/s/1JSpc9BZXFlPkXxRK4qUCyw) 代码：ol3j 

## CCPD数据集标注处理
> CCPD数据集没有专门的标注文件，每张图像的文件名就是该图像对应的数据标注。
> 
> 例如图片3061158854166666665-97_100-159&434_586&578-558&578_173&523_159&434_586&474-0_0_3_24_33_32_28_30-64-233.jpg的文件名可以由分割符'-'分为多个部分：

1. 3061158854166666665为区域（这个值可能有问题，可以不管）；
1. 97_100对应车牌的两个倾斜角度-水平倾斜角和垂直倾斜角, 水平倾斜97度, 竖直倾斜100度。水平倾斜度是车牌与水平线之间的夹角。二维旋转后，垂直倾斜角为车牌左边界线与水平线的夹角。CCPD数据集中这个参数标注可能不那么准，这个指标具体参考了论文Hough Transform and Its Application in Vehicle License Plate Tilt Correction；
1. 159&434_586&578对应边界框左上角和右下角坐标:左上(159, 434), 右下(586, 578)；
1. 558&578_173&523_159&434_586&474对应车牌四个顶点坐标(右下角开始顺时针排列)：右下(558, 578)，左下(173, 523)，左上(159, 434)，右上(586, 474)；
1. 0_0_3_24_33_32_28_30为车牌号码（第一位为省份缩写），在CCPD2019中这个参数为7位，CCPD2020中为8位，有对应的关系表；
1. 64为亮度，数值越大车牌越亮（可能不准确，仅供参考）；
1. 233为模糊度，数值越小车牌越模糊（可能不准确，仅供参考）。

对于每张图片的标注信息直接字符分割即可。一个展示CCPD数据集单张图片标注的Python代码如下。
```python
# -*- coding: utf-8 -*-
"""
下面的代码读取了CCPD中的一张图片，并绘制了其车牌的边界框，关键点，车牌名。
"""

from PIL import Image, ImageDraw, ImageFont
import os


provincelist = [
    "皖", "沪", "津", "渝", "冀",
    "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云",
    "西", "陕", "甘", "青", "宁",
    "新"]

wordlist = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9"]

# --- 绘制边界框


def DrawBox(im, box):
    draw = ImageDraw.Draw(im)
    draw.rectangle([tuple(box[0]), tuple(box[1])],  outline="#FFFFFF", width=3)

# --- 绘制四个关键点


def DrawPoint(im, points):

    draw = ImageDraw.Draw(im)

    for p in points:
        center = (p[0], p[1])
        radius = 5
        right = (center[0]+radius, center[1]+radius)
        left = (center[0]-radius, center[1]-radius)
        draw.ellipse((left, right), fill="#FF0000")

# --- 绘制车牌


def DrawLabel(im, label):
    draw = ImageDraw.Draw(im)
   # draw.multiline_text((30,30), label.encode("utf-8"), fill="#FFFFFF")
    font = ImageFont.truetype('simsun.ttc', 64)
    draw.text((30, 30), label, font=font)

# --- 图片可视化


def ImgShow(imgpath, box, points, label):
    # 打开图片
    im = Image.open(imgpath)
    DrawBox(im, box)
    DrawPoint(im, points)
    DrawLabel(im, label)
    # 显示图片
    im.show()
    im.save('result.jpg')


def main():
    # 图像路径
    imgpath = 'ccpd_green/val/0136360677083-95_103-255&434_432&512-432&512_267&494_255&434_424&449-0_0_3_25_30_24_24_32-98-218.jpg'

    # 图像名
    imgname = os.path.basename(imgpath).split('.')[0]

    # 根据图像名分割标注
    _, _, box, points, label, brightness, blurriness = imgname.split('-')

    # --- 边界框信息
    box = box.split('_')
    box = [list(map(int, i.split('&'))) for i in box]

    # --- 关键点信息
    points = points.split('_')
    points = [list(map(int, i.split('&'))) for i in points]
    # 将关键点的顺序变为从左上顺时针开始
    points = points[-2:]+points[:2]

    # --- 读取车牌号
    label = label.split('_')
    # 省份缩写
    province = provincelist[int(label[0])]
    # 车牌信息
    words = [wordlist[int(i)] for i in label[1:]]
    # 车牌号
    label = province+''.join(words)

    # --- 图片可视化
    ImgShow(imgpath, box, points, label)


if __name__ == "__main__":
    main()
```