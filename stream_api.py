from flask import Flask, render_template, Response
import cv2,time
import sys
sys.path.insert(0, './distance_detection')
sys.path.insert(0, './face_mask')
from social_distance_detector import *
from detect_mask_video import *


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('response.html')

def gen():
    cap = cv2.VideoCapture(0)

    while(cap.isOpened()):

        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (0,0), fx=0.5, fy= 0.5)
            frame = cv2.imencode('.jpg',img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'+ frame + b'\r\n')
            time.sleep(0.1)
        else:
            break

def gen_distance():
    #read the next frame from the file
    while vs.isOpened():
        (grabbed, frame) = vs.read()
    
    #if the frame was not grabbed, then we have reached the end of the stream
    
        if not grabbed:
            return

        #resize the frame and then detect people(and only people) in it
        frame = imutils.resize(frame, width = 600)
        results = detect_people(frame, net, ln, personIdx = LABELS.index("person"))
        #initialize the set of indexes that violate the minimum social distance

        violate = set()

        #ensure there are atleast two people detections (required in order to compute)
        if len(results) >= 2:
            #extract all centroids from the results and compute the euclidean distances between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids , centroids, metric="euclidean")
            #loop over the upper triangualr of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    #check to see if the distance between any two centroid pairs is less than the configured number of pixels
                    if D[i, j] < conf.MIN_DISTANCE:
                        #update the violation set with the indexes of the centroid pairs
                        violate.add(i)
                        violate.add(j)
        #loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            #extract the bounding box and centroid coordinates, then initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            #if the index pair exists within the violation set, then update the color
            if i in violate:
                color = (0, 0, 255)
            #draw (1) a bounding vbox around the person and (2) the centroid coordinates of the person
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)
        #draw the total number of social distancing violations on the output frame
        text = "Social distancing violations: {}".format(len(violate))
        # print(violate)
        cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        
        #check to see if the output frame should be displayed to out screen
        if args["display"] > 0:
            #show the output frame
            cv2.imshow("Frame", frame)
            # print(type(frame))
            frame = cv2.imencode('.jpg',frame)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'+ frame + b'\r\n')
            time.sleep(0.1)
            
            # key = cv2.waitKey(1) & 0xFF

            # #if the 'q' key was pressed, break from the loop
            # if key == ord("q"):
            #     sys.exit()   


def gen_face():
    prototxtPath = r"face_mask/face_detector/deploy.prototxt"
    weightsPath = r"face_mask/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model("face_mask/mask_detector.model")
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        # cv2.imshow("Frame", frame)
        frame = cv2.imencode('.jpg',frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'+ frame + b'\r\n')
        time.sleep(0.1)
        
    
    

@app.route('/video_feed')
def video_feed():

    return Response(gen_distance(),
    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/face_feed')
def face_feed():
    
    return Response(gen_face(),
    mimetype="multipart/x-mixed-replace; boundary=frame")

app.run(port=2000)