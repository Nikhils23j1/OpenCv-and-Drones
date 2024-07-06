import cv2
# image_path='/Users/nikhil/My Computer/A Year 2/DSA/python/image processing/Midterm/color-shapes.png'
# image_path="/Users/nikhil/My Computer/A Year 2/DSA/python/image processing/Midterm/brainy-shapes-in-bag-40pc-5-shape-4-colour-2-size.jpg"
image_path="/Users/nikhil/My Computer/A Year 2/DSA/python/image processing/Midterm/Paint-Stampers-Geometric.jpg"
def shape_detection(image_path)
img=cv2.imread(image_path)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
# contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
_,thresh=cv2.threshold(edges,200,255,cv2.THRESH_BINARY)
contours,heirarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

areas=[] #storing i, area, contour


for i,contour in enumerate(contours):
    if i==0: #here the i=0 is the complete image hence passed
        continue
        
    epsilon=0.01*cv2.arcLength(contour,True) #1%error in arc length for approximation (mayebe the squae as some irregularities  in the edges)
    approx=cv2.approxPolyDP(contour,epsilon,True) #true means contour is closed shape
    area=cv2.contourArea(contour)
    areas.append((i,area,contour))
    
    cv2.drawContours(img,contour,-1,(0,0,0),2)
    x,y,w,h =cv2.boundingRect(approx)
    x_mid=int(x+w/4)
    y_mid=int(y+h/1.5)
    coords=(x_mid,y_mid)
    colour=(0,0,0)
    font=cv2.FONT_HERSHEY_DUPLEX
    
    if len(approx)==3:
        cv2.putText(img,'Triangle',coords,font,1,colour,1)
    elif len(approx)==4:
        cv2.putText(img,'Rectangle',coords,font,1,colour,1)
    elif len(approx)==5:
        cv2.putText(img,'Pentagon',coords,font,1,colour,1)  
    elif len(approx)==6:
        cv2.putText(img,'Hexagon',coords,font,1,colour,1)
    elif len(approx)==7:
        cv2.putText(img,'Heptagon',coords,font,1,colour,1)
    elif len(approx)==8:
        cv2.putText(img,'Octagon',coords,font,1,colour,1)
    elif len(approx)==9:
        cv2.putText(img,'Nonagon',coords,font,1,colour,1)
    else:
        cv2.putText(img,'Circle',coords,font,1,colour,1)

areas.sort(key=lambda x: x[1], reverse=True)

for i in range(2):
    index, _, contour = areas[i]
    x, y, w, h = cv2.boundingRect(contour)
    colour = (255, 0, 0)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, f"Largest-{i + 1}", (int(x + w / 2), int(y + h / 1.1)), font, 1, colour, 1)
    
cv2.imshow('shapes',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
    