import cv2,selectivesearch
import numpy as np
import Intersection as union

candidates = set()
while True:
    image = cv2.imread('image/dog.337.jpg')
    img_lbl, regions = selectivesearch.selective_search(image)

    for r in regions:
        candidates.add(r['rect'])
    for x,y,w,h in candidates:
        iou = union.get_union([71, 63, 189, 199],[x,y,x+w,y+h])
        if iou > .7:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,233,12),1)
            cv2.putText(image,'Dog',(x-2,y-2),1,1,(1,22,121),1)
    cv2.imshow('',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()