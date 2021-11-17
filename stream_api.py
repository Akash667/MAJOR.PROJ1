from flask import Flask, render_template, Response, request
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from werkzeug.wrappers import request

from config import social_distancing_config as conf
from config.detection import detect_people
from scipy.spatial import distance as dist

import numpy as np
import argparse
import imutils
import time
import cv2
import os
import sys

app = Flask(__name__)

vs = cv2.VideoCapture(0+cv2.CAP_DSHOW)

@app.route('/')
def index():
    return render_template('response.html')

# def gen():
#     cap = cv2.VideoCapture(0)

#     while(cap.isOpened()):

#         ret, img = cap.read()
#         if ret == True:
#             img = cv2.resize(img, (0,0), fx=0.5, fy= 0.5)
#             frame = cv2.imencode('.jpg',img)[1].tobytes()
#             yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'+ frame + b'\r\n')
#             time.sleep(0.1)
#         else:
#             break


@app.route('/switch_to_cam')
def webcam():
    global vs
    vs.release()
    vs = cv2.VideoCapture(0+cv2.CAP_DSHOW)
    return ('', 204)

@app.route('/switch_to_file')
def fileload():
    global vs
    vs.release()
    if request.method == 'POST':
        file_path = request.form["path"]
    vs= cv2.VideoCapture(vs = cv2.VideoCapture(0+cv2.CAP_DSHOW))
    return ('', 204)

def gen_distance():
    #read the next frame from the file
    global vs
    # time.sleep(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type = str, default = "", help = "path to (optional) input video file")
    ap.add_argument("-o", "--output", type = str, default ="", help = "path to (optional) output video file")
    ap.add_argument("-d", "--display", type = int, default = 1, help = "whether the output frame should be displayed or not")
    args = vars(ap.parse_args())

    #load the coco class laels from oyt YOLO model that was trained on
    labelsPath = os.path.sep.join([conf.MODEL_PATH, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    #derive the paths to the YOLO weights and model configuration
    weights_path = os.path.sep.join([conf.MODEL_PATH, "yolov3.weights"])
    config_path = os.path.sep.join([conf.MODEL_PATH, "yolov3.cfg"])

    #load the YOLO object detector trained on coco dataset(80 classes)
    print("[INFO] loading YOLO from disk..")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    #check if we are going to use GPU
    if conf.USE_GPU:
        #set CUDA as the backend and target
        print("[INFO] setting backend and target to CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    #determine only the output layer names that we need from YOLO
    ln = net.getLayerNames()
    # print(ln)
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    #initialize the video stream and pointer to output video file
    print("[INFO] accessing video stream...")

    
    # vs = cv2.VideoCapture(0+cv2.CAP_DSHOW)


    writer = None
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
            # cv2.imshow("Frame", frame)
            # print(type(frame))
            frame = cv2.imencode('.jpg',frame)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'+ frame + b'\r\n')
            time.sleep(0.1)
    

def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	faces = []
	locs = []
	preds = []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	return (locs, preds)

def gen_face():
    # time.sleep(2)
    prototxtPath = r"face_mask/face_detector/deploy.prototxt"
    weightsPath = r"face_mask/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model("face_mask/mask_detector.model")
    print("[INFO] starting video stream...")
    # vs = cv2.VideoCapture(0+cv2.CAP_DSHOW)

    global vs

    while vs.isOpened():
        (grabbed, frame) = vs.read()

        if not grabbed:
            return
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
    # vs.release()
    return Response(gen_distance(),
    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/face_feed')
def face_feed():
    # vs.release()
    return Response(gen_face(),
    mimetype="multipart/x-mixed-replace; boundary=frame")

app.run(port=2000)


