import time
import cv2
import numpy as np
from PIL import ImageGrab
import keras
from keras.models import load_model
from keras import backend as K
from key_util import press_key, release_key


def straight():
    release_key('A')
    release_key('D')
    press_key('W')
    # time.sleep(0.1)


def left():
    release_key('D')
    press_key('A')
    # time.sleep(0.1)


def right():
    release_key('A')
    press_key('D')
    # time.sleep(0.1)


def keys_to_output(keys):
    output = [0, 0, 0]
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output


def countdown(sec):
    for i in range(sec, 0, -1):
        print(i)
        time.sleep(1)
    print('start!')


def main():
    last_time = time.time()
    frame, accum_time, fps = 0, 0, 0

    model = load_model('model.h5')
    print('Model loading complete!')
    count = 0
    while True:
        # screenshot normalization
        screen = np.array(ImageGrab.grab(bbox=(0, 0, 800, 600)))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # image process
        mask = np.zeros_like(gray)
        vertices = np.array([[0, 500], [0, 300], [200, 200], [600, 200], [800, 300], [800, 500]])
        cv2.fillPoly(mask, [vertices], 255)
        gray = cv2.bitwise_and(gray, mask)

        # gray = cv2.Sobel(gray, cv2.CV_16S, 1, 1, 5)  # Sobel operator filtering
        # gray = cv2.convertScaleAbs(gray)
        gray = cv2.Canny(gray, 50, 150)  # edges detection
        # ret, gray = cv2.threshold(qray, 160, 255, cv2.THRESH_BINARY)
        # ret, gray = cv2.threshold(
        #     gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # threshold algorithm
        # gray = cv2.GaussianBlur(gray, (3, 3), 0)
        mask = np.zeros_like(gray)
        vertices = np.array([[0, 400], [0, 300], [270, 220], [530, 220], [800, 300], [800, 400]])
        cv2.fillPoly(mask, [vertices], 255)
        gray = cv2.bitwise_and(gray, mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # kernel of closing operation
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)  # closing operation

        # detect lines
        # left_line, right_line = [], []

        # lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 100, 100, 10)
        # try:
        #     for x1, y1, x2, y2 in lines[0]:
        #         if (y2 - y1) / (x2 - x1) > 0:
        #             left_line.append((x1, y1, x2, y2))
        #         else:
        #             right_line.append((x1, y1, x2, y2))
        #         cv2.line(screen, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # except Exception:
        #     pass

        cv2.putText(screen, 'fps:{}'.format(fps), (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

        # calculate fps
        this_time = time.time()
        # print('loop took {} seconds'.format(this_time-last_time))
        accum_time += this_time - last_time
        last_time = time.time()

        frame += 1
        if accum_time >= 1:
            fps = frame
            # print('fps:', frame)
            frame, accum_time = 0, 0

        # get train image
        test_img = gray[220:400, :]
        test_img = cv2.resize(test_img, (400, 90))

        img_ndarray = np.asarray(test_img, dtype='float64') / 255
        test_data = np.ndarray.flatten(img_ndarray)
        test_data = test_data.astype('float32')

        if K.image_data_format() == 'channels_first':
            test_data = test_data.reshape(1, 1, 90, 400)
        else:
            test_data = test_data.reshape(1, 90, 400, 1)

        preds = model.predict(test_data)
        class_ = np.argmax(preds[0])
        if class_ == 0:
            left()
        elif class_ == 1:
            straight()
        else:
            right()

        count += 1

        if count == 100:
            break
        # show screenshot
        cv2.imshow('screen', test_img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destoryAllWindows()
            break


if __name__ == "__main__":
    main()
