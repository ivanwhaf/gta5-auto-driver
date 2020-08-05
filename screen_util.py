import numpy as np
from PIL import ImageGrab
import cv2
import time


def grab_screen():
    last_time = time.time()
    frame, accum_time, fps = 0, 0, 0

    while True:
        # screenshot normalization
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
        # screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.GaussianBlur(screen, (3, 3), 0)

        # calculate fps
        this_time = time.time()
        print('loop took {} seconds'.format(this_time-last_time))
        accum_time += this_time-last_time
        last_time = time.time()

        frame += 1
        if accum_time >= 1:
            fps = frame
            print('fps:', frame)
            frame, accum_time = 0, 0

        cv2.putText(screen, 'fps:{}'.format(fps), (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

        # show screenshot
        cv2.imshow('screen', screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destoryAllWindows()
            break


def main():
    grab_screen()


if __name__ == "__main__":
    main()
