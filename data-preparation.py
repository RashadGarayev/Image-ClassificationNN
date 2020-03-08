import cv2,joblib,selectivesearch
import numpy as np 
from collections import namedtuple
from skimage import io
import Intersection as union
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm, metrics
train_data = []
train_label = []
row = 128
cols = 128
channels = 3
nb_rects = 10000
counter = 0
falsecounter = 0
flag = 0
fflag = 0
bflag = 0
p = 0
n = 0

candidates = set()
Detection = namedtuple("Detection", ["image_path", "coordinate"])
examples = [
	Detection("dog.391.jpg", [13, 4, 192, 194]),
	Detection("dog.337.jpg", [71, 63, 189, 199]),
	Detection("dog.370.jpg", [48, 30, 341, 326]),
	Detection("dog.398.jpg", [48,42,491,192]),
	Detection("dog.385.jpg", [76,5,477,396]),
    Detection("dog.406.jpg", [93,27,312,421]),
    Detection("dog.409.jpg", [64,61,198,186]),
    Detection("dog.395.jpg", [66,60,496,323]),
    Detection("dog.364.jpg", [66,22,428,486]),
    Detection("dog.380.jpg", [46,5,357,484]),
    Detection("dog.394.jpg", [111,24,428,407])]
for detection in examples:
    
    image = io.imread('image/'+detection.image_path)
    im,regions = selectivesearch.selective_search(image,scale=1,sigma=1)
    for r in regions:
        if r['rect'] in candidates:
            continue
        if r['size'] < 1000:
            continue
        candidates.add(r['rect'])
        for  x, y, w, h in candidates:
            iou = union.get_union(detection.coordinate,[x,y,x+w,y+h])
            if counter < 160:
                if iou > .7:
                    PosImage = image[y:y+h,x:x+w]
                    resized = cv2.resize(PosImage, (128,128), interpolation = cv2.INTER_AREA)
                    cv2.imwrite("IMAGE/positive%i.jpg" %p, resized)
                    train_data.append(resized)
                    train_label.append(1)
                    counter+=1
                    p+=1 #for image saving
            else:
                fflag = 1
            if falsecounter <160:
                if iou<.3:
                    NegImage = image[y:y+h,x:x+w]
                    resized = cv2.resize(NegImage, (128,128), interpolation = cv2.INTER_AREA)
                    cv2.imwrite("IMAGE/negative%i.jpg" %n, resized)
                    train_data.append(resized)
                    train_label.append(0)
                    falsecounter+=1
                    n+=1
            else:
                bflag = 1
    if fflag == 1 and bflag == 1:
        
        flag = 1
print('Data Prepared........')
print('Train Data:',len(train_data))
print('Labels(0,1)',len(train_label))
print("""
Classification with SVM


""")
train_data = np.float32(train_data)
train_label = np.array(train_label)                
            
rand = np.random.RandomState(421)
shuffle = rand.permutation(len(train_data))
train_data = train_data[shuffle]
train_label = train_label[shuffle]
train_data=train_data.reshape((len(train_data),128*128*3))

X_train, X_test, y_train, y_test = train_test_split(train_data,train_label)
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
svc = SVC()
model = GridSearchCV(svc, param_grid)
print('Training...... Support Vector Machine')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Classification report for - \n{}:\n{}\n".format(model, metrics.classification_report(y_test, y_pred)))
model_name = 'model/models.sav'
joblib.dump(model, model_name)
loaded_model = joblib.load(model_name)
result = loaded_model.score(X_test, y_test)
print(result)
print('------------------------------------')
im = cv2.imread('test/dog.jpg')
im = cv2.resize(im,(128,128))
pred = model.predict(im.reshape(1,-1))
print(pred)
