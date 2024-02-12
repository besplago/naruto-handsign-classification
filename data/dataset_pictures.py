"""Create a dataset of pictures for each label in the data folder"""
import os
import sys
import cv2 as cv
from cv2 import waitKey

DATA_FOLDER: str = r"E:\Repos\naruto-handsign-classification\data"
TRAIN_DATA_FOLDER: str = os.path.join(DATA_FOLDER, "train_data")
TEST_DATA_FOLDER: str = os.path.join(DATA_FOLDER, "test_data")
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
PICTURES_PER_LABEL: int = 500
TEST_SPLIT: int = 20  # Percentage of data to be in the test set


def create_dataset() -> None:
    """Create a dataset of pictures for each label in the data folder"""
    if not os.path.exists(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)

    if not os.path.exists(TRAIN_DATA_FOLDER):
        os.makedirs(TRAIN_DATA_FOLDER)

    if not os.path.exists(TEST_DATA_FOLDER):
        os.makedirs(TEST_DATA_FOLDER)

    camera = cv.VideoCapture(0)

    if not camera.isOpened():
        print("Camera not opened, terminating..")
        sys.exit()

    for label in LABELS:
        label_train_folder = os.path.join(TRAIN_DATA_FOLDER, label)
        label_test_folder = os.path.join(TEST_DATA_FOLDER, label)
        os.makedirs(label_train_folder, exist_ok=True)
        os.makedirs(label_test_folder, exist_ok=True)

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
            cv.imshow(f"Do the {folder} sign", gray)
            gray = cv.resize(gray, (600, 400))
            if count < PICTURES_PER_LABEL * (TEST_SPLIT / 100):
                cv.imwrite(os.path.join(TEST_DATA_FOLDER, folder, f"img{count}.png"), gray)
            else:
                cv.imwrite(os.path.join(TRAIN_DATA_FOLDER, folder, f"img{count}.png"), gray)
            count = count + 1
            print(count)
            waitKey(100)

    camera.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    create_dataset()
