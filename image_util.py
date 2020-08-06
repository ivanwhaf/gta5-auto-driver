# @Author: Ivan
# @LastEdit: 2020/8/6
import os
import cv2  # install
import numpy as np  # install
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt  # installs

# ----------------Neural Network Visualization----------------

# 输入图像维度
width, height, depth = 100, 100, 3


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.transpose(1, 2, 3, 0)
    elif K.image_data_format() == 'channels_last':
        x = x.transpose(3, 1, 2, 0)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def get_intermediate_output(model, layer_name, img):
    """Get the output of intermediate layer.
    Args:
           model: keras model.
           layer_name: name of layer in the model.
           img: processed input image.
    Returns:
           intermediate_output: feature map.
    """
    try:
        # this is the placeholder for the intermediate output
        out_intermediate = model.get_layer(layer_name).output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    # get the intermediate layer model
    intermediate_layer_model = Model(
        inputs=model.input, outputs=out_intermediate)

    # get the output of intermediate layer model
    intermediate_output = intermediate_layer_model.predict(img)
    return intermediate_output[0]


def show_intermediate_output(model, layer_name, image):
    """show the output of intermediate layer.
    Args:
           model: keras model.
           layer_name: name of layer in the model.
           image: processed input image.
    Returns:
           display_grid: feature maps grid.
    """
    if depth == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (width, height))

    img_ndarray = np.asarray(image, dtype='float64') / 255
    test_data = np.ndarray.flatten(img_ndarray)
    test_data = test_data.astype('float32')

    if K.image_data_format() == 'channels_first':
        test_data = test_data.reshape(1, depth, height, width)
    else:
        test_data = test_data.reshape(1, height, width, depth)

    output = get_intermediate_output(model, layer_name, test_data)  # 中间层输出
    n = output.shape[-1]  # 特征图中特征个数
    size = output.shape[1]  # 特征图边长
    display_grid = np.zeros((size * 1, n * size))  # 网格

    for i in range(n):
        channel_image = output[:, :, i]
        display_grid[0:size, i * size:(i + 1) * size] = channel_image

    plt.figure()
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, cmap='viridis')
    plt.savefig('visualize/' + layer_name + '_output.jpg')  # 保存中间层输出图
    plt.show()  # must show after imshow

    return display_grid


def show_heatmap(model, layer_name, image):
    """show the heatmap of intermediate layer.
    Args:
           model: keras model.
           layer_name: name of layer in the model.
           image: processed input image.
    Returns:
           heatmap: the heatmap of the trained model.
           superimposed_img: heatmap apply on input image
    """
    img = image.copy()
    if depth == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (width, height))

    img_ndarray = np.asarray(image, dtype='float64') / 255
    test_data = np.ndarray.flatten(img_ndarray)
    test_data = test_data.astype('float32')

    if K.image_data_format() == 'channels_first':
        test_data = test_data.reshape(1, depth, height, width)
    else:
        test_data = test_data.reshape(1, height, width, depth)

    preds = model.predict(test_data)
    index = np.argmax(preds[0])  # 输出类别的索引
    output = model.output[:, index]

    layer = model.get_layer(layer_name)  # 中间层

    grads = K.gradients(output, layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, layer.output[0]])

    pooled_grads_value, layer_output_value = iterate([test_data])

    for i in range(layer_output_value.shape[-1]):
        layer_output_value[:, :, i
        ] *= pooled_grads_value[i]
    heatmap = np.mean(layer_output_value, axis=-1)

    # heatmap = np.maximum(heatmap, 0)
    # heatmap /= np.max(heatmap)

    plt.matshow(heatmap)
    plt.savefig('visualize/heatmap.jpg')
    plt.show()

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)  # 转换为rgb格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 热力图强度因子0.4
    cv2.imwrite('visualize/heatmap_apply.jpg', superimposed_img)

    return heatmap, superimposed_img


# -----------------------------------------------------------------------------------


# -------------------------------CV Algorithms--------------------------------------


def face_detect():
    """
    人脸检测函数
    调用摄像头实时检测出人脸并用矩形框框出
    """
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    while True:
        ret, fram = cap.read()
        faces = face_cascade.detectMultiScale(fram, 1.1, 7)
        for x, y, w, h in faces:
            cv2.rectangle(fram, (x - 5, y - 25), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('fram', fram)


def cut_face(path):
    """
    裁剪人脸函数
    检测出人脸后用新的人脸图片覆盖原图
    1.图片文件夹path目录下必须包含以每一类图片为一个文件夹的子文件夹
    如dataset文件夹下包含c1,c2,c3三个类别的子文件夹
    每个子文件夹包含相应图片,如c1文件夹下包含1.jpg,2.jpg
    2.文件夹路径名及所有文件名必须是英文
    3.文件所在根目录下必须包含haarcascade_frontalface_default.xml文件
    Args:
        path:图片文件夹路径
    Returns:
        无

    """
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img_categories = os.listdir(path)
    for img_category in img_categories:
        img_category_path = os.path.join(path, img_category)
        if os.path.isdir(img_category_path):
            imgs = os.listdir(img_category_path)
            for img in imgs:
                img_path = os.path.join(img_category_path, img)
                face = cv2.imread(img_path)  # 读取图片
                faces = face_cascade.detectMultiScale(face, 1.1, 7)  # 检测人脸
                for x, y, w, h in faces[0]:
                    # cv2.rectangle(face,(x-5,y-25),(x+w,y+h),(0,255,0),2)
                    face = face[y - 60:y + h + 15, x:x + w]
                cv2.imwrite(img_path, face)  # 用裁剪后的人脸覆盖原图片
                print(img_path + '--cuting successfully')
    print('all faces cut successfully!')


def d_hash(img):
    # 差值hash算法
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_ = ''
    for i in range(8):
        for j in range(8):
            if img[i, j] > img[i, j + 1]:
                hash_ = hash_ + '1'
            else:
                hash_ = hash_ + '0'
    print("dHash:" + str(hash_))
    return hash_


def a_hash(img):
    # 均值hash算法
    img = cv2.resize(img, (8, 8))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_ = ''
    average = 0
    for i in range(8):
        for j in range(8):
            average = average + img[i, j]
    average = average / 64

    for i in range(8):
        for j in range(8):
            if img[i, j] > average:
                hash_ = hash_ + '1'
            else:
                hash_ = hash_ + '0'
    print("aHash:" + str(hash_))
    return hash_


def p_hash(img):
    # 感知hash算法
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_ = ''
    mean = 0.0
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img
    vis1 = cv2.dct(vis0)
    for i in range(8):
        for j in range(8):
            mean += vis1[i, j]
    mean = mean / 64

    for i in range(8):
        for j in range(8):
            if vis1[i, j] >= mean:
                hash_ = hash_ + '1'
            else:
                hash_ = hash_ + '0'
    print("pHash:" + str(hash_))
    return hash_


def hamming_distance(hash1, hash2):
    # 计算两值的汉明距离
    hamming = 0
    for i in range(64):
        if hash1[i] != hash2[i]:
            hamming = hamming + 1
    return hamming


def compare_hamming_distance(img1, img2, func):
    # 比较两图的汉明距离
    if func == 'aHash':
        hamming = hamming_distance(a_hash(img1), a_hash(img2))
    elif func == 'pHash':
        hamming = hamming_distance(p_hash(img1), p_hash(img2))
    elif func == 'dHash':
        hamming = hamming_distance(d_hash(img1), d_hash(img2))
    else:
        return None
    return hamming


def blur_test():
    # 图像模糊测试函数
    img = cv2.imread('1.jpg')
    img = cv2.resize(img, (800, 1000), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(grad_x, grad_y)
    gradient = cv2.convertScaleAbs(gradient)

    gradient = cv2.blur(gradient, (3, 3))
    ret, binary = cv2.threshold(gradient, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('img', binary)
    i = cv2.waitKey(0)


def harris(img):
    # cornerHarris角点检测函数
    # img=cv2.imread('6.jpg')
    # img=cv2.resize(img,(800,800),interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow('harris', img)
    # i=cv2.waitKey(0)


def draw_faces(img):
    # 检测人脸并框出
    # img=cv2.resize(img,(800,800),interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        'D:/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('img', img)


def sift_test(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    img1 = cv2.resize(img1, (800, 800), interpolation=cv2.INTER_AREA)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray1=cv2.GaussianBlur(gray1,(5,5),0)

    img2 = cv2.resize(img2, (800, 800), interpolation=cv2.INTER_AREA)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # gray2=cv2.GaussianBlur(gray2,(5,5),0)

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # img=cv2.drawKeypoints(img,kp,img,color=(255,0,255))
    # bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # bf=cv2.BFMatcher()
    # matches=bf.knnMatch(des1,des2,k=2)
    matches = flann.knnMatch(des1, des2, k=2)
    print('matches:' + str(len(matches)), end='')
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append([m])
    print('  good:' + str(len(good)))
    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None, flags=2)
    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None, flags=2)
    # cv2.imshow('img',img3)
    # i=cv2.waitKey(0)
    cv2.imshow('img', img2)


def surf_test_fast(kp1, des1, img1, img2):
    surf = cv2.xfeatures2d.SURF_create(400)
    img2 = cv2.resize(img2, (800, 800), interpolation=cv2.INTER_AREA)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # gray2=cv2.GaussianBlur(gray2,(5,5),0)
    # surf.hessianThreshold = 500
    kp2, des2 = surf.detectAndCompute(gray2, None)
    # print(len(des1))
    # img=cv2.drawKeypoints(img,kp,img,color=(255,0,255))
    # bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # bf=cv2.BFMatcher()
    # matches=bf.knnMatch(des1,des2,k=2)
    try:
        matches = flann.knnMatch(np.asarray(
            des1, np.float32), np.asarray(des2, np.float32), k=2)
    except:
        return
    # matches=flann.knnMatch(des1,des2,k=2)
    print('matches:' + str(len(matches)), end='')
    good = []
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            good.append([m])
    print('  good:' + str(len(good)))
    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None, flags=2)
    img3 = cv2.drawMatchesKnn(img1, kp1, gray2, kp2, good, None, flags=2)
    cv2.imshow('img3', img3)


def surf_test(img1, img2):
    surf = cv2.xfeatures2d.SURF_create()
    img1 = cv2.resize(img1, (800, 800), interpolation=cv2.INTER_AREA)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray1=cv2.GaussianBlur(gray1,(5,5),0)

    img2 = cv2.resize(img2, (800, 800), interpolation=cv2.INTER_AREA)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # gray2=cv2.GaussianBlur(gray2,(5,5),0)
    # surf.hessianThreshold = 500
    kp1, des1 = surf.detectAndCompute(gray1, None)
    kp2, des2 = surf.detectAndCompute(gray2, None)
    # print(len(des1))
    # img=cv2.drawKeypoints(img,kp,img,color=(255,0,255))
    # bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # bf=cv2.BFMatcher()
    # matches=bf.knnMatch(des1,des2,k=2)
    try:
        matches = flann.knnMatch(np.asarray(
            des1, np.float32), np.asarray(des2, np.float32), k=2)
    except:
        return
    # matches=flann.knnMatch(des1,des2,k=2)
    print('matches:' + str(len(matches)), end='')
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append([m])
    print('  good:' + str(len(good)))
    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None, flags=2)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    cv2.imshow('img3', img3)
    # i=cv2.waitKey(0)


def orb_test(img1, img2):
    orb = cv2.ORB_create()

    img1 = cv2.resize(img1, (800, 800), interpolation=cv2.INTER_AREA)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)

    img2 = cv2.resize(img2, (800, 800), interpolation=cv2.INTER_AREA)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # img=cv2.drawKeypoints(img,kp,img,color=(255,0,255))
    # bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

    FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # f=cv2.BFMatcher()

    des1 = np.asarray(des1, np.float32)
    des2 = np.asarray(des2, np.float32)
    # matches=flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except:
        return
    # matches=bf.match(des1,des2)
    # matches=flann.knnMatch(des1,des2,k=2)
    print('matches:' + str(len(matches)), end='')
    good = []
    try:
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
    except:
        return
    print('  good:' + str(len(good)))
    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None, flags=2)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None, flags=2)
    cv2.imshow('img', img3)
    # i=cv2.waitKey(0)


def get_max_contour(contours):
    # 获取最大的轮廓
    max_area = 0
    max_cnt = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_cnt = cnt
    return max_cnt


def get_range_contours(contours, low, high):
    # 获取指定面积范围内的轮廓，返回轮廓列表list
    contours_list = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if low < area < high:
            contours_list.append(cnt)
    return contours_list


def draw_contours(img):
    # 画轮廓函数

    # img=cv2.resize(img,(720,1280),interpolation=cv2.INTER_AREA)
    # img= cv2.blur(img,(3,3))    #进行滤波去掉噪声
    # img= cv2.medianBlur(img,5)    #进行滤波去掉噪声
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray=img
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(50, 50))
    # 开闭运算，先开运算去除背景噪声，再继续闭运算填充目标内的孔洞
    # opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('sa',closed)

    # cv2.imshow('gray',gray)
    ret, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    # cv2.imshow('binary',binary)
    # contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours,hierarchy=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img,contours,-1,(0,0,0),cv2.FILLED)
    print(len(contours))
    if contours:
        c_max = []
        max_contour = get_max_contour(contours)
        c_max.append(max_contour)
        try:
            cv2.drawContours(img, c_max, -1, (0, 0, 255), 3)
        except Exception as e:
            return

        # r1=np.zeros(img.shape[:2],dtype="uint8")#创建黑色图像
        # 将轮廓信息转换成(x, y)坐标，并加上矩形的高度和宽度
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)  # 画出矩形

        # mask=r1
        # masked=cv2.bitwise_and(img,img,mask=mask)

        # rect = cv2.minAreaRect(max_contour)
        # box = cv2.boxPoints(rect)
        # box =np.int0(box)
        # cv2.drawContours(img, [box], 0, (0, 0, 255), 3)  # 画出该矩形
        # 注：OpenCV没有函数能直接从轮廓信息中计算出最小矩形顶点的坐标。所以需要计算出最小矩形区域，
        # 然后计算这个矩形的顶点。由于计算出来的顶点坐标是浮点型，但是所得像素的坐标值是整数（不能获取像素的一部分），
        # 所以需要做一个转换

        # (x,y),radius = cv2.minEnclosingCircle(max_contour)
        # center = (int(x),int(y))
        # radius = int(radius)
        # img = cv2.circle(img,center,radius,(0,255,0),2)

        # cv2.imshow('final',masked)
        cv2.imshow('final', img)
    # i=cv2.waitKey(0)


def get_video():
    # 获取视频帧测试函数
    cap = cv2.VideoCapture(0)
    # img1=cv2.imread('1.jpg')
    # img1=cv2.imread('5.jpg')
    # surf = cv2.xfeatures2d.SURF_create(400)
    # img1=cv2.resize(img1,(800,800),interpolation=cv2.INTER_AREA)
    # gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    # kp1,des1=surf.detectAndCompute(gray1,None)

    # mog2=cv2.createBackgroundSubtractorMOG2()
    # mog2=cv2.bgsegm.createBackgroundSubtractorGMG()
    while True:
        # time.sleep(0.034)
        ret, frame = cap.read()
        if not ret:
            continue
        # hamming=compare_hamming_distance(img1,frame,'dHash')
        # print(hamming)
        # frame=cv2.blur(frame,(3,3))
        # frame=cv2.GaussianBlur(frame,(5,5),0)
        # fgmask=mog2.apply(frame)
        # draw_contours(fgmask)
        # cv2.imshow('1',fgmask)
        draw_contours(frame)
        # surf_test_fast(kp1,des1,img1,frame)
        # draw_faces(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# -------------------------------------------------------------


def main():
    # get_video()
    pass


if __name__ == '__main__':
    main()
