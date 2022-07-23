import cv2
import numpy as np
import socket
import json
import easyocr
from openvino.runtime import Core, Layout, Type
from openvino.preprocess import PrePostProcessor, ColorFormat
from utils.augmentations import letterbox
from config import (
    MODEL_PATH,
    SERVER_ADDR,
    SEND_COOR_FLAG,
    CLASS_NAMES,
    COLORS,
    CAMERA_INDEX,
    FORMULA_NAMES,
)
import torch
from coordinate import pixel_to_world

window_name = "Frame"


def get_net(model_path):
    core = Core()
    model = core.read_model(model_path)
    ppp = PrePostProcessor(model)

    # Declare input data information:
    ppp.input().tensor().set_color_format(ColorFormat.BGR).set_element_type(
        Type.u8
    ).set_layout(Layout("NHWC"))

    # Specify actual model layout
    ppp.input().model().set_layout(Layout("NCHW"))

    # Set output tensor information:
    # - precision of tensor is supposed to be 'f32'
    ppp.output().tensor().set_element_type(Type.f32)

    # Apply preprocessing modifing the original 'model'
    # - Precision from u8 to f32
    # - color plane from BGR to RGB
    # - subtract mean
    # - divide by scale factor
    # - Layout conversion will be done automatically as last step
    ppp.input().preprocess().convert_element_type(Type.f32).convert_color(
        ColorFormat.RGB
    ).mean([0.0, 0.0, 0.0]).scale([255.0, 255.0, 255.0])

    # Integrate preprocessing steps into model and compile Model
    print(f"Build preprocessor: {ppp}")
    model = ppp.build()
    net = core.compile_model(model, "AUTO")
    return net


def get_result(predictions):
    class_ids = []
    confidences = []
    boxes = []

    for pred in predictions:
        for i, det in enumerate(pred):
            confidence = det[4]
            scores = det[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > 0.25:
                confidences.append(float(confidence))
                class_ids.append(class_id)
                x, y, w, h = (
                    det[0].item(),
                    det[1].item(),
                    det[2].item(),
                    det[3].item(),
                )
                left = int((x - 0.5 * w - dw) / ratio[0])
                top = int((y - 0.5 * h - dh) / ratio[1])
                width = int(w / ratio[0])
                height = int(h / ratio[1])
                box = np.array([(left), top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.45)

    filtered_ids = []
    filered_confidences = []
    filtered_boxes = []
    print(indexes)
    for i in indexes:
        print(i)
        print(i[0])
        filtered_ids.append(class_ids[i[0]])
        filered_confidences.append(confidences[i[0]])
        filtered_boxes.append(boxes[i[0]])

    return filtered_ids, filered_confidences, filtered_boxes


def show_box(frame, filtered_ids, filered_confidences, filtered_boxes):
    # Show box
    for (class_id, _, box) in zip(filtered_ids, filered_confidences, filtered_boxes):
        color = COLORS[int(class_id) % len(COLORS)]
        cv2.rectangle(frame, box, color, 2)
        cv2.rectangle(
            frame, (box[0], box[1] - 10), (box[0] + box[2], box[1]), color, -1
        )
        world_coor = pixel_to_world(box[0] + box[2] / 2, box[1] + box[3])
        cv2.putText(
            frame,
            "{}: ({:.2f} {:.2f} {:.2f})".format(
                CLASS_NAMES[class_id], world_coor[0], world_coor[1], world_coor[2]
            ),
            (box[0], box[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
    return frame


def get_camera(window_name, width, height):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(3, width)
    cap.set(4, height)
    cv2.namedWindow(window_name)
    return cap


def send_coor(s, filtered_ids, filtered_boxes):
    datas = {"requestType": 2, "id": 1, "type": "camera", "data": []}
    for (class_id, box) in zip(filtered_ids, filtered_boxes):
        world_coor = pixel_to_world(box[0] + box[2] / 2, box[1] + box[3])
        # Convert to unity world coor system: Y-up, left handed coordinates (We just swap y and z here)
        datas["data"].append(
            {
                "name": CLASS_NAMES[class_id],
                "coor": (world_coor[0], world_coor[2], world_coor[1]),
            }
        )
    s.sendall((json.dumps(datas) + "\n").encode())


def send_coor_with_ocr(s, filtered_ids, filtered_boxes, ocr_match):
    datas = {"requestType": 3, "id": 1, "type": "camera", "data": []}
    index = 0
    print("len(ocr_match)  ", len(ocr_match))
    for (class_id, box) in zip(filtered_ids, filtered_boxes):
        text = ""
        for i in range(0, len(ocr_match)):
            if ocr_match[i][1] == index:
                text = ocr_match[i][0]
                break
        world_coor = pixel_to_world(box[0] + box[2] / 2, box[1] + box[3])
        # Convert to unity world coor system: Y-up, left handed coordinates (We just swap y and z here)
        datas["data"].append(
            {
                "name": CLASS_NAMES[class_id],
                "coor": (world_coor[0], world_coor[2], world_coor[1]),
                "text": text,
            }
        )
        index += 1
    s.sendall((json.dumps(datas) + "\n").encode())


def edit_distance(s1, s2):
    n = len(s1)
    m = len(s2)

    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            # print(dp)
            if i == n:
                dp[i][j] = m - j
            elif j == m:
                dp[i][j] = n - i
            else:
                d1 = 1 + dp[i + 1][j]
                d2 = 1 + dp[i][j + 1]
                d3 = dp[i + 1][j + 1] if s1[i] == s2[j] else 1 + dp[i + 1][j + 1]
                dp[i][j] = min(d1, d2, d3)
    return dp[0][0]


def select_formula(s):
    min_distance = 100
    result = ""
    for formula in FORMULA_NAMES:
        distance = edit_distance(s, formula)
        if edit_distance(s, formula) < min_distance:
            min_distance = distance
            result = formula

    return result


if __name__ == "__main__":
    net = get_net(MODEL_PATH)
    cap = get_camera(window_name, 1280, 720)
    reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory

    if SEND_COOR_FLAG:
        s = socket.socket()
        s.connect(SERVER_ADDR)

    while True:
        ret, frame = cap.read()
        letterbox_img, ratio, (dw, dh) = letterbox(frame, auto=False)
        # Change shape from HWC to NHWC
        input_tensor = np.expand_dims(letterbox_img, axis=0)

        predictions = net([input_tensor])[net.outputs[0]]

        filtered_ids, filered_confidences, filtered_boxes = get_result(predictions)

        print("ocr_result:   ", )
        ocr_result = reader.readtext(frame, allowlist='0123456789QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm')  # Using OCR to parse the images
        print(ocr_result)
        ocr_match = []
        for result in ocr_result:
            # x = result[0][0][0]
            # y = result[0][0][1]
            # w = result[0][2][0] - result[0][0][0]
            # h = result[0][2][1] - result[0][0][1]
            # left = int((x - 0.5 * w - dw) / ratio[0])
            # top = int((y - 0.5 * h - dh) / ratio[1])
            # width = int(w / ratio[0])
            # height = int(h / ratio[1])
            left = result[0][0][0]
            top = result[0][0][1]
            width = result[0][2][0] - result[0][0][0]
            height = result[0][2][1] - result[0][0][1]
            cv2.rectangle(
                frame, (int(result[0][0][0]), int(result[0][0][1])), (int(result[0][2][0]),
                                                                      int(result[0][2][1])), (0, 0, 255), -1
            )
            index = 0
            for (class_id, box) in zip(filtered_ids, filtered_boxes):
                index += 1
                if class_id != 56:
                    continue
                if left >= box[0] and top >= box[1] and left + width <= box[0] + box[2] and top + height <= box[1] + \
                        box[3]:
                    tmp = [select_formula(result[1]), index - 1]
                    ocr_match.append(tmp)
        print("ocr_match:")
        print(ocr_match)

        if SEND_COOR_FLAG:
            # send_coor(s, filtered_ids, filtered_boxes)
            send_coor_with_ocr(s, filtered_ids, filtered_boxes, ocr_match)

        frame = show_box(frame, filtered_ids, filered_confidences, filtered_boxes)

        cv2.imwrite("window_name.jpg", frame)
        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            break

    cv2.destroyAllWindows()
    if SEND_COOR_FLAG:
        s.close()
