import argparse
import json

import cv2

DETECTOR = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))


def detect(image):
    h, w = image.shape[:2]
    corners, ids, _ = DETECTOR.detectMarkers(image)
    if ids is None:
        return {}
    markers = {}
    for i, c in zip(ids.flatten(), corners):
        centre = c[0].mean(axis=0) / [w, h]
        markers[int(i)] = centre.round(4).tolist()
    return markers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="image file, video file, or camera index")
    source = ap.parse_args().source
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        print(json.dumps(detect(frame), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
