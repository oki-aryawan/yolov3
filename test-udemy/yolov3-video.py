import cv2
import numpy as np
import time

video = cv2.VideoCapture('videos/cattle.mp4')
writer = None
h, w = None, None

with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]

network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                     'yolov3.weights')
layers_name_all = network.getLayerNames()
layers_names_output = \
    [layers_name_all[i - 1] for i in network.getUnconnectedOutLayers()]
probability_min = 0.5
threshold = 0.3
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
f = 0
t = 0

while True:
    ret, frame = video.read()
    if not ret:
        break
    if w is None or h is None:
        h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    network.setInput(blob)
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()
    f += 1
    f += end - start
    print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))

    bounding_boxes = []
    confidence = []
    class_number = []

    for result in output_from_network:
        for detected_obj in result:
            scores = detected_obj[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]

            if confidence_current > probability_min:
                box_current = detected_obj[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_w, box_h = box_current
                x_min = int(x_center - (box_w / 2))
                y_min = int(y_center - (box_h / 2))
                bounding_boxes.append([x_min, y_min, int(box_w), int(box_h)])
                confidence.append(float(confidence_current))
                class_number.append(class_current)

    result = cv2.dnn.NMSBoxes(bounding_boxes, confidence,
                              probability_min, threshold)
    if len(result) > 0:
        for i in result.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_w, box_h = bounding_boxes[i][2], bounding_boxes[i][3]
            colour_box_current = colours[class_number[i]].tolist()

            cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_w, y_min + box_h),
                          colour_box_current, 2)

            text_box_current = '{}: {:.4f}'.format(labels[int(class_number[i])],
                                                   confidence[i])

            cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('videos/result-catte.mp4', fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)
    writer.write(frame)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# print(('Total number of frames', f))
# print('Total amount of time {:.5f} seconds'.format(t))


# print('FPS: ', round((f / t), 1))

video.release()
writer.release()
cv2.destroyAllWindows()
