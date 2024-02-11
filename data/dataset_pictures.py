"""Create a dataset of pictures for each label in the data folder"""
import os
import sys
import cv2 as cv
from cv2 import waitKey

DATA_FOLDER: str = r"E:\Repos\naruto-handsign-classification\data"
LABELS: list = [
    "bird",
    "boar",
    "dog",
    "dragon",
    "hare",
    "horse",
    "monkey",
    "ox",
    "ram",
    "rat",
    "snake",
    "tiger"
]
PICTURES_PER_LABEL: int = 50

def create_dataset() -> None:
    """Create a dataset of pictures for each label in the data folder"""
    if not os.path.exists(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)

    camera = cv.VideoCapture(0)

    if not camera.isOpened():
        print("Camera not opened, terminating..")
        sys.exit()


    for label in LABELS:
        if not os.path.exists(label):
            os.mkdir(f"{DATA_FOLDER}/{label}")

    for folder in LABELS:
        print("Press 'x' for " + folder)
        userinput = input()
        if userinput != 'x':
            print("Wrong Input, try again")
            continue

        waitKey(2000)

        count = 0

        while count < PICTURES_PER_LABEL:
            status, frame = camera.read()

            if not status:
                print("Frame is not been captured, terminating...")
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            cv.imshow("Video Window",gray)
            gray = cv.resize(gray, (600,400))
            cv.imwrite(f"{DATA_FOLDER}/{folder}/img{count}.png", gray)
            count = count + 1
            print(count)
            waitKey(100)

    camera.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    create_dataset()
