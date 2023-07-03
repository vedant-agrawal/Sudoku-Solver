# Importing Dependencies
from util import *
import solver
from keras.utils import img_to_array

# Intializing the image path, dimensions and the pre-trained CNN model
path = "sudoku.jpg"
height = 810
width = 810
model = initializePredictionModel()

# Reading the image and making copies that will be used later
img = cv.imread(path)
img = cv.resize(img, (height, width))
imgContour = img.copy()
imgBoard = img.copy()

# Preprocessing the image
pImg = preProcess(img)

# Finding contours on the image and drawing them on the imgContour
contours, hierarchy = cv.findContours(pImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(imgContour, contours, -1, (0, 0, 255), 3)

# Isolating the board and getting it's location in the image
imgBoard, location = isolateBoard(contours, imgBoard)

# Converting the isolated board to grayscale
grayBoard = cv.cvtColor(imgBoard, cv.COLOR_BGR2GRAY)

#Extracting each cell from the board
boxes = extract_digits(grayBoard)

# Predicting the digits of the board
board = []
for box in boxes:
    if box is None:
        board.append(0)
    else:
        box = cv.resize(box, (28,28))
        box = box.astype('float32') / 255.0
        box = img_to_array(box)
        box = np.expand_dims(box, axis=0)
        
        
        prediction = model.predict(box)[0]
        board.append(np.argmax(prediction) + 1)
 
# Converting the board to a 2-D numpy array
solveBoard = np.array(board).astype('uint8').reshape(9,9)

# SOlving and s=overlaying the solution
if solver.solve(solveBoard):
    bin = np.where(np.array(board)>0, 0, 1)
    numbers = solveBoard.flatten() * bin
    mask = np.zeros_like(imgBoard)
    overlay = display(mask, numbers)
    inv = transfromPerspectiveInv(overlay, location)
    combined = cv.addWeighted(img, 0.7, inv, 1, 0)
    cv.imshow("Final_result", combined)
    cv.waitKey(0)
    cv.imwrite('result.jpg', combined)
else:
    print("No solution. Error in prediction by model")