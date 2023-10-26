import cv2


# Create our body classifier
fullbody_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    fullbody = fullbody_cascade.detectMultiScale(gray,1.1,5)
    
    # Extract bounding boxes for any bodies identified
    for(x,y,w,h) in fullbody:
       cv2.rectangle(frame,(x,y),(x+w,y+h),2)

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
