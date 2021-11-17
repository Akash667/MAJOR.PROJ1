#base path to yolo directory
MODEL_PATH = "distance_detection/yolo-coco"

#initializing minimum probability to filter weak detections along with the
#threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

#boolean value to indicate if NIVIDIA CUDA GPU should be used
USE_GPU = True

#The minimum safe distance(in pixels) that two people can be safe from
#each other
MIN_DISTANCE = 50
