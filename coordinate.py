from unittest import result
import numpy as np
import cv2
from config import intrinsic_path, R_path, T_path

window_name = "Frame"
camera_intrinsic = np.mat(np.loadtxt(intrinsic_path))
r = np.loadtxt(R_path)
t = np.asmatrix(np.loadtxt(T_path)).T


def __pixel_to_world(camera_intrinsics, r, t, img_points):
    K_inv = camera_intrinsics.I
    R_inv = np.asmatrix(r).I
    R_inv_T = np.dot(R_inv, np.asmatrix(t))
    world_points = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_point in img_points:
        coords[0] = img_point[0]
        coords[1] = img_point[1]
        coords[2] = 1.0
        cam_point = np.dot(K_inv, coords)
        cam_R_inv = np.dot(R_inv, cam_point)
        scale = R_inv_T[2][0] / cam_R_inv[2][0]
        scale_world = np.multiply(scale, cam_R_inv)
        world_point = np.asmatrix(scale_world) - np.asmatrix(R_inv_T)
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0] = world_point[0]
        pt[1] = world_point[1]
        pt[2] = 0
        world_points.append(pt.T.tolist())

    return world_points


def pixel_to_world(x, y):
    result = __pixel_to_world(camera_intrinsic, r, t, [[x, y]])
    return (result[0][0][0], result[0][0][1], result[0][0][2])


def get_camera(window_name, width, height):
    cap = cv2.VideoCapture(1)
    cap.set(3, width)
    cap.set(4, height)
    cv2.namedWindow(window_name)
    return cap


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        result = __pixel_to_world(camera_intrinsic, r, t, [[x, y]])
        print(
            "({}, {}) --> ({}, {}, {}))".format(
                x, y, result[0][0][0], result[0][0][1], result[0][0][2]
            )
        )


if __name__ == "__main__":
    cap = get_camera(window_name, 1280, 720)
    while True:
        ret, frame = cap.read()

        cv2.setMouseCallback(window_name, on_EVENT_LBUTTONDOWN)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
