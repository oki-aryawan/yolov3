import numpy as np
import cv2
import time


def yolov3(path):
    img_bgr = cv2.imread(path)
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(img_bgr, 1/255.0, (416, 416), swapRB=True,
                                 crop=False)
    with open('yolo-coco-data/coco.names') as f:
        labels = [line.strip() for line in f]

    network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                         'yolo-coco-data/yolov3.weights')
    layers_name_all = network.getLayerNames()
    layer_names_out = \
        [layers_name_all[i - 1] for i in network.getUnconnectedOutLayers()]
    prob = 0.5
    thresh = 0.3

    colours = np.random.randint(9, 255, size=(len(labels), 3), dtype='uint8')

    network.setInput(blob)
    start = time.time()
    out_from_net = network.forward(layer_names_out)
    end = time.time()

    bounding_box = []
    confidence = []
    class_number = []

    for result in out_from_net:
        for detected_obj in result:
            scores = detected_obj[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]

            if confidence_current > prob:
                box_current = detected_obj[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_windth, box_height = box_current
                x_min = int(x_center - (box_windth / 2))
                y_min = int(y_center - (box_height / 2))

                bounding_box.append([x_min, y_min, int(box_windth), int(box_height)])
                confidence.append(float(confidence_current))
                class_number.append(class_current)

    result = cv2.dnn.NMSBoxes(bounding_box, confidence, prob, thresh)
    counter = 1
    if len(result) > 0:
        for i in result.flatten():
            counter += 1
            x_min, y_min = bounding_box[i][0], bounding_box[i][1]
            box_windth, box_height = bounding_box[i][2], bounding_box[i][3]
            color_box_current = colours[class_number[i]].tolist()

            cv2.rectangle(img_bgr, (x_min, y_min), (x_min + box_windth, y_min + box_height),
                          color_box_current, 2)
            text_box_current = '{}: {:.4f}'.format(labels[int(class_number[i])],
                                                   confidence[i])
            cv2.putText(img_bgr, text_box_current, (x_min, y_min -5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, color_box_current, 2)
            cv2.imwrite('result.jpg', img_bgr)

