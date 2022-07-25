import cv2
import numpy as np
import socket
import json
import easyocr
import threading
import sys
import copy
from openvino.runtime import Core, Layout, Type
from openvino.preprocess import PrePostProcessor, ColorFormat
from utils.augmentations import letterbox
from config import (
    ENABLE_OCR,
    MODEL_PATH,
    SERVER_ADDR,
    SEND_COOR_FLAG,
    CLASS_NAMES,
    COLORS,
    CAMERA_INDEX,
    FORMULA_NAMES,
    WHITE_LIST,
    SEND_COOR_WITH_OCR_FLAG,
    OCR_LIST,
)
import torch
from coordinate import pixel_to_world

window_name = "Frame"

shared_struct = {}
mutex = threading.Lock()
event = threading.Event()


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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.45)

    filtered_ids = []
    filered_confidences = []
    filtered_boxes = []
    for i in indexes:
        filtered_ids.append(class_ids[i[0]])
        filered_confidences.append(confidences[i[0]])
        filtered_boxes.append(boxes[i[0]])

    return filtered_ids, filered_confidences, filtered_boxes


def show_box(frame, filtered_ids, filered_confidences, filtered_boxes):
    # Show box
    for (class_id, confidences, box) in zip(
        filtered_ids, filered_confidences, filtered_boxes
    ):
        if CLASS_NAMES[class_id] not in WHITE_LIST:
            continue
        color = COLORS[int(class_id) % len(COLORS)]
        cv2.rectangle(frame, box, color, 2)
        cv2.rectangle(
            frame, (box[0], box[1] - 10), (box[0] + box[2], box[1]), color, -1
        )
        world_coor = pixel_to_world(box[0] + box[2] / 2, box[1] + box[3])
        cv2.putText(
            frame,
            "{}: ({:.2f} {:.2f} {:.2f} {:.2f})".format(
                CLASS_NAMES[class_id],
                world_coor[0],
                world_coor[1],
                world_coor[2],
                confidences,
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
        if CLASS_NAMES[class_id] in OCR_LIST:
            continue
        world_coor = pixel_to_world(box[0] + box[2] / 2, box[1] + box[3])
        # Convert to unity world coor system: Y-up, left handed coordinates (We just swap y and z here)
        datas["data"].append(
            {
                "name": CLASS_NAMES[class_id],
                "coor": (world_coor[0], world_coor[2], world_coor[1]),
            }
        )
    mutex.acquire()
    s.sendall((json.dumps(datas) + "\n").encode())
    mutex.release()


def send_coor_with_ocr(s, filtered_ids, filtered_boxes, ocr_match):
    datas = {"requestType": 3, "id": 1, "type": "camera", "data": []}
    index = 0
    print("len(ocr_match): {}".format(len(ocr_match)))
    for (class_id, box) in zip(filtered_ids, filtered_boxes):
        if CLASS_NAMES[class_id] not in OCR_LIST:
            continue
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
    mutex.acquire()
    s.sendall((json.dumps(datas) + "\n").encode())
    mutex.release()


def edit_distance(s1, s2):
    n = len(s1)
    m = len(s2)

    dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
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
    min_distance = 3
    result = ""
    for formula in FORMULA_NAMES:
        distance = edit_distance(s, formula)
        if distance <= min_distance:
            min_distance = distance
            result = formula

    return result


def start_ocr():
    while True:
        event.wait()
        mutex.acquire()
        frame, ids, boxes = (
            shared_struct["frame"],
            shared_struct["ids"],
            shared_struct["boxes"],
        )
        mutex.release()

        print("ocr_result:   ",)
        ocr_result = reader.readtext(
            frame,
            allowlist="0123456789QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm",
        )  # Using OCR to parse the images
        print(ocr_result)
        ocr_match = []
        for result in ocr_result:
            left = result[0][0][0]
            top = result[0][0][1]
            right = result[0][2][0]
            bottom = result[0][2][1]
            cv2.rectangle(
                frame,
                (int(result[0][0][0]), int(result[0][0][1])),
                (int(result[0][2][0]), int(result[0][2][1])),
                (0, 0, 255),
                2,
            )
            for index, (class_id, box) in enumerate(zip(ids, boxes)):
                if class_id != 56:
                    continue
                if (
                    left >= box[0]
                    and top >= box[1]
                    and right <= box[0] + box[2]
                    and bottom <= box[1] + box[3]
                ):
                    formula = select_formula(result[1])
                    if len(formula) == 0:
                        continue
                    ocr_match.append([formula, index])
        print("ocr_match: {}".format(ocr_match))
        cv2.imwrite("{}_ocr.jpg".format(window_name), frame)
        if SEND_COOR_WITH_OCR_FLAG:
            send_coor_with_ocr(s, ids, boxes, ocr_match)
        key = cv2.waitKey(1)
        if key == 27:
            break


if __name__ == "__main__":
    net = get_net(MODEL_PATH)
    cap = get_camera(window_name, 1280, 720)
    reader = easyocr.Reader(
        ["en"]
    )  # this needs to run only once to load the model into memory

    if SEND_COOR_FLAG or SEND_COOR_WITH_OCR_FLAG:
        s = socket.socket()
        s.connect(SERVER_ADDR)

    if ENABLE_OCR:
        t = threading.Thread(target=start_ocr, daemon=True)
        t.start()

    while True:
        ret, frame = cap.read()
        letterbox_img, ratio, (dw, dh) = letterbox(frame, auto=False)
        # Change shape from HWC to NHWC
        input_tensor = np.expand_dims(letterbox_img, axis=0)

        predictions = net([input_tensor])[net.outputs[0]]

        filtered_ids, filtered_confidences, filtered_boxes = get_result(predictions)

        if SEND_COOR_FLAG:
            send_coor(s, filtered_ids, filtered_boxes)

        mutex.acquire()
        shared_struct["frame"] = copy.deepcopy(frame)
        shared_struct["ids"] = copy.deepcopy(filtered_ids)
        shared_struct["confidences"] = copy.deepcopy(filtered_confidences)
        shared_struct["boxes"] = copy.deepcopy(filtered_boxes)
        mutex.release()

        frame = show_box(frame, filtered_ids, filtered_confidences, filtered_boxes)

        event.set()

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            break

    cv2.destroyAllWindows()
    if SEND_COOR_FLAG or SEND_COOR_WITH_OCR_FLAG:
        s.close()
    torch.cuda.empty_cache()
    sys.exit()
