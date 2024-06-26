INTRINSIC_PATH = "camera_matrix.txt"
R_PATH = "./rotM.txt"
T_PATH = "./translation_vector.txt"

MODEL_PATH = "models/yolov5s.onnx"

SERVER_ADDR = ("127.0.0.1", 10002)
SEND_COOR_FLAG = False
SEND_COOR_WITH_OCR_FLAG = False

ENABLE_OCR = False

CAMERA_INDEX = 1

CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

WHITE_LIST = [
    "person",
    "chair",
    "umbrella",
]

# items with ocr
OCR_LIST = [
    "chair",
]

COLORS = [
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 255),
    (120, 156, 67),
    (62, 179, 128),
    (214, 109, 169),
]

FORMULA_NAMES = [
    "NaCl",
    "HNO3",
    "K2S",
    "NaOH",
    "Na2S",
    "KCl",
    "CaCl2",
    "BaCl2",
    "H2SO4",
    "MgCl2",
    "KOH",
    "He",
    "Ne",
]
