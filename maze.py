import math
import cv2
import numpy as np


treasure = []
cap_size = 10
maze_size = 19
photo_number = 3
lower_green = np.array([45, 50, 50])
upper_green = np.array([75, 255, 255])
lower_yellow = np.array([13, 100, 100])
upper_yellow = np.array([30, 255, 255])
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([125, 255, 255])
lower_red1 = np.array([0, 100, 100])  # red 1
upper_red1 = np.array([8, 255, 255])
lower_red2 = np.array([170, 100, 100])  # red 2
upper_red2 = np.array([180, 255, 255])


def turn_90(photo_place):
    # 逆时针90
    # 左下             左上           右上            右下
    # 读取原始图像
    src_image = cv2.imread(photo_place)
    # 构造旋转矩阵
    center = (src_image.shape[1] / 2, src_image.shape[0] / 2)
    M = cv2.getRotationMatrix2D(center, 90, 1)

    # 应用旋转变换
    rotated = cv2.warpAffine(src_image, M, (src_image.shape[1], src_image.shape[0]))
    # 显示结果
    # cv2.imshow("Rotated Image", rotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("maze_reshaped_90.jpg", rotated)


def detect_circle(pic):
    count_all_all = 1
    size = 120
    while count_all_all == 1:
        # 二值化处理
        ret, pic = cv2.threshold(
            pic, size, 255, cv2.THRESH_BINARY
        )  # _INV | cv2.THRESH_OTSU)[1]
        count_all = 1
        ksize = (5, 5)
        count_more_1 = 3
        while count_all == 1:
            pic = cv2.GaussianBlur(pic, ksize, 0)  ### 高斯滤波
            # cv2.imshow("result", pic)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite("test.png", pic)

            key_param2 = 15

            count = 1
            count_more = 0
            median_1 = []
            while count == 1:
                arr = []
                idx = []
                circles = cv2.HoughCircles(
                    pic,
                    cv2.HOUGH_GRADIENT,
                    1,
                    10,
                    param1=200,
                    param2=key_param2,
                    minRadius=0,
                    maxRadius=30,
                )  ### 返回三个值，圆心x ，圆心y ，半径
                circles = circles[0, :, :]
                circles = np.uint16(np.around(circles))

                # 对识别到的圆形进行检测
                for i in range(len(circles)):
                    arr.append(circles[i][2])
                # print(arr)
                # 剔除异常值
                median = np.median(arr)
                median_1.append(median)

                # 防止中间值随整体数值变大而变大
                mean_median = np.mean(median_1)
                if median - mean_median > 2:
                    median = mean_median
                    median_1.pop()

                # print(median)
                z_scores = arr - median
                idx = np.where(np.abs(z_scores) > 3)[0]
                # print(idx)
                if len(idx) > 0:
                    for i in range(len(idx)):
                        row_idx = idx[i] - i
                        circles[row_idx:-1] = circles[row_idx + 1 :]
                        circles = circles[:-1]
                # print("剩余" + str(len(circles)))

                # 对异常情况处理
                if count_more > 8 and count_more_1 == 3:
                    ksize = (3, 3)
                    count = 0
                    count_more_1 = 1
                if count_more > 8 and count_more_1 == 1:
                    ksize = (1, 1)
                    count = 0

                if count_more_1 == 1 and count_more > 8:
                    count = 0
                    count_all = 0
                    size = size - 10
                    print("-10")

                # 判断参数下一步如何调整
                if len(circles) < 8:
                    key_param2 = key_param2 - 1
                elif len(circles) > 8:
                    key_param2 = key_param2 + 1
                    count_more = count_more + 1
                else:  # 退出所有循环
                    count = 0
                    count_all = 0
                    count_all_all = 0
                    # print(circles)
                    return circles


def color_find_red(img):
    # 转为HSL色彩空间
    hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 定义红色区间
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    # 使用inRange函数进行阈值分割，提取对应颜色区域
    mask_red = cv2.inRange(hsl, lower_red, upper_red)
    mask_red2 = cv2.inRange(hsl, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red, mask_red2)
    # 使用 findContours 函数查找红色矩形的轮廓
    contours, hierarchy = cv2.findContours(
        mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # 遍历红色矩形的轮廓，获取其坐标信息
    coordinates_red = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 20:  # 过滤掉面积太小的轮廓
            #           cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            coordinates_red.append([x, y, w, h])  # 获取的是矩形左上角坐标已经长和宽
            coordinates_red = np.array(coordinates_red)
    # print(coordinates_red)
    # 显示结果图像
    # cv2.imshow("red", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return coordinates_red


def color_find_red_1(img):
    # 转为HSL色彩空间
    hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 定义红色区间
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    # 使用inRange函数进行阈值分割，提取对应颜色区域
    mask_red = cv2.inRange(hsl, lower_red, upper_red)
    mask_red2 = cv2.inRange(hsl, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red, mask_red2)
    # 使用 findContours 函数查找红色矩形的轮廓
    contours, hierarchy = cv2.findContours(
        mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # 遍历红色矩形的轮廓，获取其坐标信息
    coordinates_red = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 1 and h > 1:  # 过滤掉面积太小的轮廓
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            coordinates_red.append([x, y, w, h])  # 获取的是矩形左上角坐标已经长和宽
            coordinates_red = np.array(coordinates_red)
    # print(coordinates_red)
    # 显示结果图像
    # cv2.imshow("red", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return coordinates_red


def color_find_blue(img):
    # 将图像转换为HSV颜色空间
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 定义要提取的颜色范围
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    # 提取蓝色区域的掩膜
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    # 对掩膜进行形态学处理，去除噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    # 使用 findContours 函数查找红色矩形的轮廓
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # 遍历红色矩形的轮廓，获取其坐标信息
    coordinates_blue = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 20:  # 过滤掉面积太小的轮廓
            #            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            coordinates_blue.append([x, y, w, h])  # 获取的是矩形左上角坐标已经长和宽
            coordinates_blue = np.array(coordinates_blue)
    # print(coordinates_blue)
    # 显示结果图像
    # cv2.imshow("blue", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return coordinates_blue


def color_find_blue_1(img):
    # 将图像转换为HSV颜色空间
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 定义要提取的颜色范围
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    # 提取蓝色区域的掩膜
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    # 对掩膜进行形态学处理，去除噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    # 使用 findContours 函数查找红色矩形的轮廓
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # 遍历红色矩形的轮廓，获取其坐标信息
    coordinates_blue = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 1 and h > 1:  # 过滤掉面积太小的轮廓
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            coordinates_blue.append([x, y, w, h])  # 获取的是矩形左上角坐标已经长和宽
            coordinates_blue = np.array(coordinates_blue)
    # print(coordinates_blue)

    # 显示结果图像
    # cv2.imshow("blue", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return coordinates_blue

    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    approx_triangles = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 3:
            approx_triangles.append(approx)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.drawContours(img, [contour], 0, (0, 255, 0), 2)
            cv2.putText(
                img,
                "Green triangle",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            print("There have Green trianle")
            return 1, approx_triangles
    return 2, approx_triangles


def cap_maze():

    # 读取校正后的迷宫图像
    while True:
        print(key_code)
        src = cv2.imread("maze_photo.jpg".format(key_code))
        # 转换为灰度图像
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)  ### 高斯滤波
        red = color_find_red(src)
        blue = color_find_blue(src)

        if blue[0][0] > red[0][0]:  # blue above
            blue[0][0] = blue[0][0] + blue[0][2]
            red[0][1] = red[0][1] + red[0][2]
            blue[0][0] += 5
            blue[0][1] -= 5
            red[0][0] -= 5
            red[0][1] += 5
        else:
            red[0][0] = red[0][0] + red[0][2]
            blue[0][1] = blue[0][1] + blue[0][2]
            red[0][0] += 5
            red[0][1] -= 5
            blue[0][0] -= 5
            blue[0][1] += 5

        west_down = np.zeros((1, 2))
        west_up = np.zeros((1, 2))
        east_down = np.zeros((1, 2))
        east_up = np.zeros((1, 2))

        west_down[0][0] = red[0][0]
        west_down[0][1] = red[0][1]
        west_up[0][0] = red[0][0]
        west_up[0][1] = blue[0][1]
        east_up[0][0] = blue[0][0]
        east_up[0][1] = blue[0][1]
        east_down[0][0] = blue[0][0]
        east_down[0][1] = red[0][1]

        # 设置sortPoints
        sortPoints = np.zeros((4, 2), dtype=np.int32)
        sortPoints[0][0] = west_down[0][0]  # 左下
        sortPoints[0][1] = west_down[0][1]
        sortPoints[1][0] = west_up[0][0]  # 左上
        sortPoints[1][1] = west_up[0][1]
        sortPoints[2][0] = east_up[0][0]  # 右上
        sortPoints[2][1] = east_up[0][1]
        sortPoints[3][0] = east_down[0][0]  # 右下
        sortPoints[3][1] = east_down[0][1]

        # 利用sortPoints的四个点进行透视变换
        minSide = min(src.shape[:2])
        center_x = src.shape[0] // 2
        center_y = src.shape[1] // 2

        srcTri = np.array(
            [sortPoints[0], sortPoints[1], sortPoints[2], sortPoints[3]],
            dtype=np.float32,
        )
        dstTri = np.array(
            [
                (center_x - 0.45 * minSide, center_y - 0.45 * minSide),
                (center_x + 0.45 * minSide, center_y - 0.45 * minSide),
                (center_x + 0.45 * minSide, center_y + 0.45 * minSide),
                (center_x - 0.45 * minSide, center_y + 0.45 * minSide),
            ],
            dtype=np.float32,
        )

        perspImage = np.zeros(
            (src.shape[0], src.shape[1], src.shape[2]), dtype=np.uint8
        )
        # 提取图像映射
        transmtx = cv2.getPerspectiveTransform(srcTri, dstTri)

        perspImage = cv2.warpPerspective(src, transmtx, perspImage.shape[:2])

        # cv2.imshow("result5", perspImage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite("maze_reshaped.jpg", perspImage)

        turn_90("maze_reshaped.jpg")

        # 读取图像
        src = cv2.imread("maze_reshaped_90.jpg")
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (3, 3), 0)  ### 高斯滤波

        # 创建FAST角点检测器
        fast = cv2.FastFeatureDetector_create()

        # 检测角点
        keypoints = fast.detect(img, None)

        # 二值化处理
        ret, thresh = cv2.threshold(
            img, 120, 255, cv2.THRESH_BINARY
        )  # _INV | cv2.THRESH_OTSU)[1]

        # 获取迷宫矩阵
        grid_size = 1  # 设置网格大小
        maze_matrix = np.zeros_like(thresh, dtype=np.int64)
        # for contour in contours:
        for y in range(0, maze_matrix.shape[0], grid_size):
            for x in range(0, maze_matrix.shape[1], grid_size):
                if thresh[y, x] == 0:  # 像素点为黑色
                    maze_matrix[y, x] = 1
        print(maze_matrix[0][0])
        # 保存为文本
        np.savetxt("matrix.txt", maze_matrix.astype(np.int64), fmt="%d")

        # 绘制角点
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

        # 获取角点坐标
        points = []
        for kp in keypoints:
            points.append(kp.pt)

        # 检测每个角点的像素值
        west_up = 10000
        west_up_array = np.zeros((1, 2))  # (y,x)
        west_down = 10000
        west_down_array = np.zeros((1, 2))
        east_up = 10000
        east_up_array = np.zeros((1, 2))
        east_down = 10000
        east_down_array = np.zeros((1, 2))

        h, w = img.shape
        # 先判断点是否为黑色，然后再判断到边界点的距离，并记录下距离最小点的坐标
        for p in points:
            x, y = p  # 点在矩阵中的坐标为(y,x)
            x = int(x)
            y = int(y)
            if (
                maze_matrix[y, x] == 1
            ):  # 检测该点为0即黑色, maze_matrix[w, h] = 1  # maze_matrix 为像素矩阵
                distance_wu = math.sqrt((y - 0) ** 2 + (x - 0) ** 2)
                distance_wd = math.sqrt((h - y) ** 2 + (x - 0) ** 2)
                distance_eu = math.sqrt((y - 0) ** 2 + (w - x) ** 2)
                distance_ed = math.sqrt((h - y) ** 2 + (w - x) ** 2)
                if distance_wu < west_up:
                    west_up = distance_wu
                    west_up_array[0][0] = y
                    west_up_array[0][1] = x
                if distance_wd < west_down:
                    west_down = distance_wd
                    west_down_array[0][0] = y
                    west_down_array[0][1] = x
                if distance_eu < east_up:
                    east_up = distance_eu
                    east_up_array[0][0] = y
                    east_up_array[0][1] = x
                if distance_ed < east_down:
                    east_down = distance_ed
                    east_down_array[0][0] = y
                    east_down_array[0][1] = x

        red = color_find_red(src)
        blue = color_find_blue(src)

        if blue[0][0] > red[0][0]:  # blue above
            red[0][0] = red[0][0] + red[0][2]
            red[0][1] = red[0][1] + red[0][2]
            east_up_array[0][0] = blue[0][1]
            east_up_array[0][1] = blue[0][0]
            west_down_array[0][0] = red[0][1]
            west_down_array[0][1] = red[0][0]
        else:
            blue[0][0] = blue[0][0] + blue[0][2]
            blue[0][1] = blue[0][1] + blue[0][2]
            east_up_array[0][0] = red[0][1]
            east_up_array[0][1] = red[0][0]
            west_down_array[0][0] = blue[0][1]
            west_down_array[0][1] = blue[0][0]

        # 显示图像
        # cv2.imshow("Image with keypoints", img_with_keypoints)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite("keypoints.jpg", img_with_keypoints)

        # 设置sortPoints
        sortPoints = np.zeros((4, 2), dtype=np.int32)
        sortPoints[0][0] = west_down_array[0][1]
        sortPoints[0][1] = west_down_array[0][0]
        sortPoints[1][0] = west_up_array[0][1]
        sortPoints[1][1] = west_up_array[0][0]
        sortPoints[2][0] = east_up_array[0][1]
        sortPoints[2][1] = east_up_array[0][0]
        sortPoints[3][0] = east_down_array[0][1]
        sortPoints[3][1] = east_down_array[0][0]

        # 利用sortPoints的四个点进行透视变换
        minSide = min(thresh.shape[:2])
        center_x = thresh.shape[0] // 2
        center_y = thresh.shape[1] // 2

        srcTri = np.array(
            [sortPoints[0], sortPoints[1], sortPoints[2], sortPoints[3]],
            dtype=np.float32,
        )
        dstTri = np.array(
            [
                (center_x - 0.45 * minSide, center_y - 0.45 * minSide),
                (center_x + 0.45 * minSide, center_y - 0.45 * minSide),
                (center_x + 0.45 * minSide, center_y + 0.45 * minSide),
                (center_x - 0.45 * minSide, center_y + 0.45 * minSide),
            ],
            dtype=np.float32,
        )

        perspImage = np.zeros(
            (src.shape[0], src.shape[1], src.shape[2]), dtype=np.uint8
        )
        # 提取图像映射
        transmtx = cv2.getPerspectiveTransform(srcTri, dstTri)

        # binImage = cv2.bitwise_not(thresh)  # 二值图像反色
        perspImage = cv2.warpPerspective(src, transmtx, perspImage.shape[:2])

        print(sortPoints)
        # 透视变换后的四个角点坐标
        perspPoints = cv2.perspectiveTransform(
            np.array([sortPoints], dtype=np.float32), transmtx
        )
        print(perspPoints)
        # perspPoints[0][i][0] and [0][i][1]
        # cv2.imshow("result5", perspImage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite("maze_reshaped.jpg", perspImage)

        # 0 west_up
        # 1 east_up
        # 2 east_down
        # 3 west_down

        w = perspPoints[0][1][0] - perspPoints[0][0][0]
        h = perspPoints[0][3][1] - perspPoints[0][0][1]
        #    print(w)
        #    print(h)
        w_per = w / 10
        h_per = h / 10
        #    print(w_per)
        #    print(h_per)

        src = cv2.imread("maze_reshaped.jpg")
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # img = cv2.GaussianBlur(img, (5, 5), 0)        ### 高斯滤波
        # img = cv2.medianBlur(img, 5)
        # img = cv2.bilateralFilter(img, 9, 75, 75)
        circles = detect_circle(img)

        src = cv2.imread("maze_reshaped.jpg")
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        for i in range(len(circles)):
            cv2.circle(
                src, (int(circles[i][0]), int(circles[i][1])), 25, (0, 0, 255), -1
            )
        cv2.imshow("result", src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("circles_points.jpg", src)

        count = 1  # 1 go on  0 go back detect_circles

        for i in range(len(circles)):
            cv2.circle(
                src, (int(circles[i][0]), int(circles[i][1])), 25, (0, 0, 255), -1
            )
            circles[i][0] = (circles[i][0] - perspPoints[0][0][0]) / w_per
            circles[i][1] = (circles[i][1] - perspPoints[0][0][1]) / h_per
            print(circles[i])

        # 判断是否有重复 or 位于临近一格
        for i in range(len(circles)):
            for j in range(i, len(circles)):
                if i == j:
                    continue
                w_count = abs(int(circles[i][0]) - int(circles[j][0]))
                h_count = abs(int(circles[i][1]) - int(circles[j][1]))
                length = w_count + h_count
                if length < 2:
                    count = 0
                    break
            if count == 0:
                break
            x = circles[i][0]
            y = circles[i][1]
            treasure.append([x, y])
        # 在每一个不同的模块识别判断后加入go back 指令，节省时间
        if count == 0:
            if key_code == photo_number:
                print("error")
                break
            print("go back1")
            key_code = key_code + 1
            continue

        for i in range(len(treasure)):
            # for j in range(len(possible_treasure)):
            x = treasure[i][0]
            y = treasure[i][1]
            element = [9 - x, 9 - y] in treasure
            if element == False:

                count == 0
                break

        if count == 0:
            if key_code == photo_number:
                print("error")
                break
            print("go back2")
            key_code = key_code + 1
            continue
        break

def maze_loader():
    matrix = np.loadtxt("matrix.txt")
    print(matrix.shape[0])
    matrix = matrix[
        :: matrix.shape[0] // (cap_size * maze_size),
        :: matrix.shape[1] // (cap_size * maze_size),
    ]
    # np.savetxt("matrix_downsampled.txt", matrix, fmt="%d")
    # 找出迷宫的实际区域，即非0元素出现的范围
    nonzero_rows = np.any(matrix == 1, axis=1)
    row_indices = np.where(nonzero_rows)[0]
    min_row, max_row = row_indices[0], row_indices[-1]
    
    min_col = np.nonzero(matrix[min_row,:])[0][0]
    max_col = np.nonzero(matrix[max_row,:])[0][-1]

    # 再次提取实际迷宫区域，即非0元素中所包围的0元素的范围
    maze_region = matrix[min_row : max_row + 1, min_col : max_col + 1]

    nonzero_rows = np.any(maze_region == 0, axis=1)
    row_indices = np.where(nonzero_rows)[0]
    min_row, max_row = row_indices[0], row_indices[-1]

    min_col = np.where(maze_region[min_row, :] == 0)[0][0]
    max_col = np.where(maze_region[max_row, :] == 0)[0][-1]

    maze_region = maze_region[min_row : max_row + 1, min_col : max_col + 1]
    # np.savetxt("maze_region.txt", maze_region.astype(np.int64), fmt="%d")

    rows, cols = maze_region.shape
    size =  max(rows,cols)
    step = size // 10
    if step % 2 == 0:
        size -= size%10
    else:
        step += 1
        size = step * 10
    maze_region = cv2.resize(maze_region.astype(np.float32), (size, size), 
                        interpolation=cv2.INTER_NEAREST).astype(np.int64)
    # np.savetxt("maze_region1.txt", maze_region.astype(np.int64), fmt="%d")
    
    # 除去宝藏点（噪声），通过将迷宫分成10*10区域进行采样
    half_step = step // 2
    print(step)
    zero_positions = []
    visit_positions = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(half_step, size, step):
        for j in range(half_step, size, step):
            if maze_region[i, j] == 1:
                zero_positions.append((i, j))
                maze_region[i, j] = 0
                while zero_positions:
                    x, y = zero_positions.pop()
                    for dx, dy in directions:
                        ni, nj = x + dx, y + dy
                        if maze_region[ni, nj] == 1:
                            maze_region[ni, nj] = 0
                            zero_positions.append((ni, nj))
                        
    # np.savetxt("maze_region2.txt", maze_region.astype(np.int64), fmt="%d")
    
    # 重构迷宫，通过将迷宫分成19*19区域进行采样
    target_maze = np.zeros((19, 19), dtype=int)
    for i in range(19):
        for j in range(19):
            x = i * half_step + half_step
            y = j * half_step + half_step
            print(x,y)
            grid = maze_region[
                x - 2 : x + 2,
                y - 2 : y + 2,
            ]
            if np.any(grid == 1):
                target_maze[i, j] = 1

    np.savetxt("maze.txt", target_maze.astype(np.int64), fmt="%d")

    # print("转换完成，结果已保存到maze.txt")


if __name__ == "__main__":
    maze_loader()
