import os,cv2
import nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

ROWS = 128
COLS = 128
CHANNELS = 1
TRAIN_DIR = 'IMAGE/'
TEST_DIR = 'TEST/'

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]


train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    
def prepare_data(images):
    m = len(images)
    X = np.zeros((m, ROWS, COLS), dtype=np.uint8)
    y = np.zeros((1, m))
    for i, image_file in enumerate(images):
        X[i,:] = read_image(image_file)
        if 'positive' in image_file.lower():
            y[0, i] = 1
        elif 'negative' in image_file.lower():
            y[0, i] = 0
    return X, y
train_set_x, train_set_y = prepare_data(train_images)
test_set_x, test_set_y = prepare_data(test_images)

train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0], ROWS*COLS*CHANNELS).T
test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0], -1).T
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255



NN = nn.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 10000, learning_rate = 0.003, print_loss= True)
im = cv2.imread('test/dog1.jpg',0)
im = cv2.resize(im,(ROWS,COLS))
test= im.reshape(1, ROWS*COLS).T
pred = nn.predict(NN["w"], NN["b"], test)
print(pred)
learning_rates = [0.001,0.01,0.005]
models = {}
for i in learning_rates:
    print("learning rate is: ",i)
    models[i] = nn.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 10000, learning_rate = i, print_loss= True)
    print("---------------------------------------------------------")


for i in learning_rates:
    plt.plot(np.squeeze(models[i]["loss"]), label= str(models[i]["learning_rate"]))
plt.ylabel('Loss')
plt.xlabel("iterations (hundreds)")
legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
