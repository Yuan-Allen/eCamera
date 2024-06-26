import cv2
import numpy as np
from config import INTRINSIC_PATH, R_PATH, T_PATH, CAMERA_INDEX

object_points = []
image_points = []
camera_matrix = np.loadtxt(INTRINSIC_PATH)
dist_coeffs = np.zeros((4, 1))


def get_camera(window_name, width, height):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(3, width)
    cap.set(4, height)
    cv2.namedWindow(window_name)
    return cap


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global image_points
    global object_points
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        image_points.append((float(x), float(y)))
        world_x, world_y, world_z = input("world coor: ").split()
        object_points.append((float(world_x), float(world_y), float(world_z)))


def get_object_points(cap):
    while True:
        ret, frame = cap.read()

        cv2.setMouseCallback("Frame", on_EVENT_LBUTTONDOWN)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break


if __name__ == "__main__":
    cap = get_camera("Frame", 1280, 720)
    get_object_points(cap)
    if len(object_points) >= 4:
        (_, rotation_vector, translation_vector, _) = cv2.solvePnPRansac(
            np.array(object_points), np.array(image_points), camera_matrix, dist_coeffs
        )
        rotM = cv2.Rodrigues(rotation_vector)[0]
        np.savetxt(R_PATH, rotM)
        np.savetxt(T_PATH, translation_vector)
        print(rotM)
        print(translation_vector)
    else:
        print("Number of points can't be less than 4.")
