# This file contains the utility functions used in main.py

# Importing Dependencies
import cv2 as cv
import numpy as np
from skimage.segmentation import clear_border
from imutils.perspective import four_point_transform
import imutils
from keras.models import load_model


def preProcess(img):
    """
    Preprocesses an image by converting it to grayscale and applying blur and thresholding to it
    :param
        img (2-d array): an image
    :return:
        thresh (2-d array) : result of converting img to grayscale and applying Gaussianblur and adaptiveThreshold on it
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (3, 3), 1)
    thresh = cv.adaptiveThreshold(blurred, 255, 1, 1, 11, 2)
    return thresh

def transformPerspective(img, location):
    """
    Returns the bird-eye view of the the piece of image represented by the location
    :param
        img (2-d array): an image
        location : location tuple of the board in the image
    :return:
        board (2-d array): bird's eye view of the image represented by the location
    """
    board = four_point_transform(img, location.reshape(4, 2))
    return cv.resize(board, (810, 810))

def isolateBoard(contours, ogImg):
    """
    Returns the board and its location in the image
    :param
        contours: list of contours encountered in the image
        ogImg : original image (sudoku.jpg)
    :return:
        result (2-d array): an image (the board)
        location: the location of the board in the Image
    """
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]
    for c in contours:
        app = cv.approxPolyDP(c, 15, True)
        if len(app) == 4:
            location = app
            break
    result = transformPerspective(ogImg, location)
    return result, location

def splitBoxes(board):
    """
    Returns a list containing the images of each cell of the board
    :param
        board (2-d array): an image (the board)
    :return:
        boxes: list containing the images of each cell of the board
    """

    rows = np.vsplit(board, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for c in cols:
            boxes.append(c)
    return boxes

def initializePredictionModel():
    """
    Initializes the pre-tained model and returns it
    :param
    :return:
        model(object): pre-tained model
    """
    model = load_model('myModel.h5')
    return model

def extract_digits(imgBoard):
    """
    Splits the board into cells and processes the images of each cell so they can be predicted
    :param
        imgBoard (2-d array): an image (the board)
    :return:
        digits: list containing the images of processed cells of the board that can be predicted by the model
    """
    digits = []
    boxes = splitBoxes(imgBoard)
    for cell in boxes:
        thresh = cv.threshold(cell, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        thresh = clear_border(thresh)
        contour = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour = imutils.grab_contours(contour)    
        if len(contour) == 0:
             digits.append(None)

        else:     
            c = max(contour, key=cv.contourArea)
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv.drawContours(mask, [c], -1, 255, -1)
            large = max(contour, key=cv.contourArea)
            (h, w) = thresh.shape
            percentFilled = cv.countNonZero(mask) / float(w * h)
            
            if percentFilled < 0.03:
                digits.append(None)
            else:
                cv.drawContours(mask, [large], -1, 255, -1)
                digits.append(thresh)  

    return digits     

def display(img, numbers, color = (130,200,130)):
    """
    Returns an image (mask) with the missing numbers displayed 
    :param
        img (2-d array): an image (the background)
        numbers: an array (flattened board) with the numbers that are already given replaced by 0's.
        color: color of the displayed numbers
    :return:
        img: the image wuth the missing numbers displayed
    """
    width = int(img.shape[1]/9)
    height = int(img.shape[0]/9)
    for i in range (9):
        for j in range (9):
            if numbers[(j*9)+i] != 0:
                cv.putText(img, str(numbers[(j*9)+i]),(i * width+ int(width / 2)-int((width / 4)), int((j + 0.6) * height)), cv.FONT_HERSHEY_COMPLEX, 2, color, 2)
    return img

def transfromPerspectiveInv(mask, location, height = 810, width = 810):
    """
    Returns an image with mask re - transformed from bird's eye view to the original view in the image
    :param 
        mask: image to be re-transformed
        location: location to re-transform to
    :return:
        result: image with the mask re-transformed to the desired location
    """
    initial = np.float32([[width, 0], [0, 0], [0, height], [width, height]])
    final = np.float32([location[0], location[1], location[2], location[3]])

    transformMat = cv.getPerspectiveTransform(initial, final)
    result = cv.warpPerspective(mask, transformMat, (height, width))
    return result


