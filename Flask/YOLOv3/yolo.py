import numpy as np
import cv2
class YOLOv3():
    def __init__(self, thr, conf, classes_to_detect, net, use):
        ''' use: 'tracking' or just 'detection' 
        YOLOv3 TRACKER
        '''
        self.thr_param = thr
        self.conf_param = conf
        self.classes_to_detect = classes_to_detect
        self.net = net
        self.use = use

    def if_there_classes(self,classID):
        flag = False
        for desiredID in self.classes_to_detect:
            if classID == desiredID:
                flag = True
        return flag    

    def detect_objects(self, image):    
        # load our input image and grab its spatial dimensions
        (H, W) = image.shape[:2]

        # determine only the *output* layer names that we need from YOLO
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (256, 192), # (96, 96) \ (192, 192) \ (256, 256) \ (384, 384)
            swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(ln)

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.conf_param and self.if_there_classes(classID):
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_param,
            self.thr_param)
        new_boxes_id = []
        # ensure at least one detection exists
        if len(idxs) > 0:

            # loop over the indexes we are keeping
            # building a list or centers we're keeping
            new_boxes = []
            if self.use == 'tracking':
                for i in idxs.flatten():
                    r = np.random.choice(255)
                    g = np.random.choice(255)
                    b = np.random.choice(255)
                    color = (r,g,b)
                    boxes[i].append(color)
                    new_boxes.append(boxes[i])
                    new_boxes_id.append(i)
            elif self.use == 'detection':
                LABELS = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
                COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")
                new_boxes = []
                new_boxes_id = []           
                for i in idxs.flatten():
                    new_boxes = []
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw a bounding box rectangle and label on the frame
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                            confidences[i])
                    cv2.putText(image, text, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    new_boxes.append(boxes[i])
                    new_boxes_id.append(i)
        return new_boxes, new_boxes_id, image                    