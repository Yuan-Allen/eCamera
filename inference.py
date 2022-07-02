import cv2
import numpy as np
from openvino.runtime import Core, Layout, Type
from openvino.preprocess import PrePostProcessor, ColorFormat
from utils.augmentations import letterbox
from config import model_path, window_name, class_names, colors


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
                confidences.append(confidence)
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
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    filtered_ids = []
    filered_confidences = []
    filtered_boxes = []

    for i in indexes:
        filtered_ids.append(class_ids[i])
        filered_confidences.append(confidences[i])
        filtered_boxes.append(boxes[i])

    return filtered_ids, filered_confidences, filtered_boxes


def show_box(frame, filtered_ids, filered_confidences, filtered_boxes):
    # Show box
    for (class_id, _, box) in zip(filtered_ids, filered_confidences, filtered_boxes):
        color = colors[int(class_id) % len(colors)]
        cv2.rectangle(frame, box, color, 2)
        cv2.rectangle(
            frame, (box[0], box[1] - 10), (box[0] + box[2], box[1]), color, -1
        )
        cv2.putText(
            frame,
            class_names[class_id],
            (box[0], box[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        print(class_names[class_id], box)
    return frame


def get_camera(window_name, width, height):
    cap = cv2.VideoCapture(1)
    cap.set(3, width)
    cap.set(4, height)
    cv2.namedWindow(window_name)
    return cap


if __name__ == "__main__":
    net = get_net(model_path)
    cap = get_camera(window_name, 1280, 720)
    while True:
        ret, frame = cap.read()

        letterbox_img, ratio, (dw, dh) = letterbox(frame, auto=False)
        # Change shape from HWC to NHWC
        input_tensor = np.expand_dims(letterbox_img, axis=0)

        predictions = net([input_tensor])[net.outputs[0]]

        filtered_ids, filered_confidences, filtered_boxes = get_result(predictions)

        frame = show_box(frame, filtered_ids, filered_confidences, filtered_boxes)

        print("Detections: " + str(len(filtered_ids)))

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
