import cv2
import numpy as np
from xgboost import XGBClassifier
from time import sleep
import os
import pickle


def get_first_frame(capture):
    pressed = -1
    image = None
    while (pressed < 0) or (pressed == 255):
        succ, source_image = capture.read(0)
        if succ:
            image = source_image.copy()
            cv2.putText(source_image, 'Please Take a Photo!', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            cv2.imshow('Say "Cheese!"', source_image)
            pressed = cv2.waitKey(10)
    if image is None:
        raise Exception('Frame is None')
    cv2.destroyAllWindows()
    image = cv2.blur(image, (3, 3))
    return image


"""
    https://stackoverflow.com/questions/37177811/crop-rectangle-returned-by-minarearect-opencv-python
"""
def cut_hand_from_mask(img, rect, box):
    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    angle = rect[2]
    if angle < -45:
        angle += 90

    # Center of rectangle in source image
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    # Size of the upright rectangle bounding the rotated rectangle
    size = (x2 - x1, y2 - y1)
    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
    # Cropped upright rectangle
    cropped = cv2.getRectSubPix(img, size, center)
    cropped = cv2.warpAffine(cropped, M, size)
    croppedW = H if H > W else W
    croppedH = H if H < W else W
    # Final cropped & rotated rectangle
    return cv2.getRectSubPix(cropped, (int(croppedW), int(croppedH)), (size[0] / 2, size[1] / 2))


def get_hand_img(first_frame, source_image, display=False, debug_boxes=False, debug_contours=False,
                 contour_min_size=5000, debug_result=False):
    diff = cv2.absdiff(first_frame, source_image)
    # diff = first_frame - source_image
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = cv2.blur(diff, (6, 6))
    cv2.imshow('Grey', diff)
    # mask = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
    _, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)  # np.zeros(image.shape, dtype=np.uint8)
    # BGR CONVERSION AND THRESHOLD

    # Do contour detection on skin region
    image, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_box = (0, 0, 0, 0)  # x, y, w, h
    max_rot_box, max_rect = None, None
    for i, cnt in enumerate(contours):
        rect = cv2.minAreaRect(cnt)
        box = np.int0(cv2.boxPoints(rect))
        x, y, w, h = cv2.boundingRect(cnt)
        if max_box[2] * max_box[3] < w * h:
            max_box = (x, y, w, h)
            max_rot_box = box
            max_rect = rect
            # cv2.rectangle(source_image, (x,y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.drawContours(source_image, [box], 0, (0, 0, 255), 2)

    x, y, w, h = max_box

    if debug_boxes:
        cv2.rectangle(source_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(source_image, [max_rot_box], 0, (0, 0, 255), 2)

    if debug_contours:
        # Draw the contour on the source image
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area > contour_min_size:
                cv2.drawContours(source_image, contours, i, (0, 255, 0), 2)

    if display:
        cv2.imshow('Camera Output', source_image)

    """if (w > 0) and (h > 0):
        res = mask[y: y + h, x: x + w]
        if debug_result:
            cv2.imshow('Hand', res)
        return res"""
    if max_rect is not None:
        res = cut_hand_from_mask(mask, max_rect, max_rot_box)
        cv2.imshow('Hand', res)
        return res
    else:
        return np.zeros((10, 10), dtype=np.uint8)


def rescale_image(tres, scale_size):
    if tres.shape[1] > tres.shape[0]:
        tres = cv2.resize(tres, (scale_size, int(scale_size*tres.shape[0]/tres.shape[1])))
    else:
        tres = cv2.resize(tres, (int(scale_size*tres.shape[1]/tres.shape[0]), scale_size))
    res = np.zeros((scale_size, scale_size), dtype=np.uint8)
    res[0:tres.shape[0], 0:tres.shape[1]] = tres
    return res


def get_train_dataset(capture, label, first_frame, scale_size=64, dataset_size=500, append=True):
    print('Getting train dataset...')
    start = 0
    if append:
        for file in os.listdir('./train'):
            if file[0] == label:
                ind = int(file[2:-4])
                if ind > start:
                    start = ind
    pressed = -1
    while (pressed < 0) or (pressed == 255):
        succ, source_image = capture.read(0)
        if succ:
            cv2.putText(source_image, label, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Press any key to start"', source_image)
            pressed = cv2.waitKey(10)
        else:
            print('(get_train_dataset_1) Not succ')
            break

    cv2.destroyAllWindows()

    key_pressed = -1  # -1 indicates no key pressed
    print('OK')

    for i in range(dataset_size):
        succ, source_image = capture.read(0)
        if succ:
            tres = get_hand_img(first_frame, source_image, display=True, debug_result=True, debug_boxes=True)
            res = rescale_image(tres, scale_size)
            cv2.imwrite('train/' + label + '_' + str(i + start) + '.png', res)
        cv2.waitKey(10)

    cv2.destroyAllWindows()


def load_train_dataset():
    x_train, y_train = [], []
    for file in os.listdir('./train'):
        img = cv2.imread('train/'+file, cv2.IMREAD_GRAYSCALE)
        #cv2.imshow('Test', img)
        #cv2.waitKey(1)
        x_train.append(img.reshape(img.shape[0]*img.shape[1]))
        #print(img.shape)
        y_train.append(ord(file[0]))
    #cv2.destroyAllWindows()
    return np.array(x_train), np.array(y_train)


def init_cls(load_model=False, save_model=True):
    cls = None
    if load_model:
        cls = pickle.load(open("model.pickle", "rb"))
        print('Model Loaded!')
    else:
        cls = XGBClassifier(n_estimators=100)
        x_train, y_train = load_train_dataset()
        print('Fitting...')
        print(x_train.shape, y_train.shape)
        cls.fit(x_train, y_train)
        if save_model:
            pickle.dump(cls, open("model.pickle", "wb"))
            print('Model saved!')
    return cls


if __name__ == '__main__':
    cls = init_cls(load_model=True)

    # Get pointer to video frames from primary device
    video_frame = cv2.VideoCapture(0)
    sleep(1)
    video_frame.read(0)

    first_frame = get_first_frame(video_frame)

    # Process the video frames
    key_pressed = -1  # -1 indicates no key pressed

    init_done = True

    while (key_pressed < 0) or (key_pressed == 255):  # any key pressed has a value >= 0

        # Grab video frame, decode it and return next video frame
        succ, source_image = video_frame.read(0)

        if succ:
            source_image = cv2.blur(source_image, (3, 3))
            res = rescale_image(
                get_hand_img(first_frame, source_image, display=False, debug_result=False, debug_contours=True,
                             debug_boxes=True), 64)
            cv2.imshow('Resized', res)
            if init_done:
                gest = cls.predict(np.array([res.reshape(res.shape[0] * res.shape[1])]))[0]
                cv2.putText(source_image, chr(gest), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Source', source_image)
            # Check for user input to close program
            key_pressed = cv2.waitKey(33)  # wait 1 milisecond in each iteration of while loop

    # Close window and camera after exiting the while loop
    cv2.destroyAllWindows()

    print(key_pressed)

    if (key_pressed != 32) and (key_pressed != 18) and (key_pressed != 13):
        get_train_dataset(video_frame, chr(key_pressed).upper(), first_frame)

    video_frame.release()