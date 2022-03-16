import cv2
import numpy as np
import time

image_bgr = cv2.imread('images/test4.jpeg')
print('Image shape: ', image_bgr.shape)
h, w = image_bgr.shape[:2]
print('Image height ={0} and width={1}'.format(h,w))

cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

blob = cv2.dnn.blobFromImage(image_bgr, 1/225.0, (416, 416),
                             swapRB=True, crop=False)
show_blob = blob[0, :, :, :].transpose(1,2,0)
print(show_blob.shape)
cv2.namedWindow('Blob Image', cv2.WINDOW_NORMAL)
cv2.imshow('Blob Image', cv2.cvtColor(show_blob, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]
print(f'List with labels name: {labels}')
network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                     'yolov3.weights')
layers_name_all = network.getLayerNames()
print(layers_name_all)
layers_name_output = [layers_name_all[i - 1] for i in network.getUnconnectedOutLayers()]
print(layers_name_output)
probability_minimum = 0.5
threshold = 0.3
colours = np.random.randint(1, 255, size=(len(labels), 3), dtype='uint8')
print(type(colours))
print(colours.shape)
print(colours[0])

network.setInput(blob)
start = time.time()
output_from_network = network.forward(layers_name_output)
end = time.time()
print('Objects Detection took {:.5f} seconds'.format(end - start))

bounding_boxes = []
confidence = []
class_number = []

for result in output_from_network:
    for detected_obj in result:
        scores = detected_obj[5:]
        class_current = np.argmax(scores)
        confidence_current = scores[class_current]

        if confidence_current > probability_minimum:
            box_current = detected_obj[0:4] * np.array([w, h, w, h])
            x_center, y_center, box_width, box_height = box_current
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))

            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidence.append(float(confidence_current))
            class_number.append(class_current)

result = cv2.dnn.NMSBoxes(bounding_boxes, confidence, probability_minimum,
                          threshold)

counter = 1

if len(result) > 0:
    for i in result.flatten():
        print('Object {0}: {1}'.format(counter, labels[int(class_number[i])]))
        counter += 1
        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

        colour_box_current = colours[class_number[i]].tolist()

        cv2.rectangle(image_bgr, (x_min, y_min),
                      (x_min + box_width, y_min + box_height),
                      colour_box_current, 2)
        text_box_current = '{}: {:.4f}'.format(labels[int(class_number[i])],
                                               confidence[i])
        cv2.putText(image_bgr, text_box_current, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

print('Total object been detected: ', len(bounding_boxes))
print('Number of object left after non-maximum suppression: ', counter - 1)
cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.imshow('Detection', image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()





