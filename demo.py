"""Demo for the model."""
import os
import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("./VGG_Naruto_Model")

LENIENCY = 100

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Reading in label images
label_images = {}
for img_name in os.listdir('images'):
    label_images.update({img_name[:-4]: cv2.imread(f'images/{img_name}')})

while True:
    success, image = cap.read()
    image = cv2.flip(image,1)
    try:
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except cv2.error:
        continue
    results = hands.process(imageRGB)

    # checking whether a hand is detected
    if results.multi_hand_landmarks:
        xs = []
        ys = []
        for handLms in results.multi_hand_landmarks: # working with each hand

            for _, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                xs.append(cx)
                ys.append(cy)

        if len(xs) > 0 and len(ys) > 0:
            ma_x = max([0, max(xs) + LENIENCY])
            ma_y = max([0, max(ys) + LENIENCY])
            mi_x = max([0, min(xs) - LENIENCY])
            mi_y = max([0, min(ys) - LENIENCY])

            image = cv2.rectangle(
                img=image, pt1=(mi_x, mi_y), pt2=(ma_x, ma_y), color=(0, 0, 255), thickness=2
            )
            cropped = image[mi_y:ma_y, mi_x:ma_x]
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            # thresholding
            tf_img = tf.image.convert_image_dtype(cropped, tf.dtypes.uint8)
            #gray = tf.squeeze(tf_img,2)
            gray = tf_img
            #thresholded = otsu_thresholding(gray)

            cropped = cv2.cvtColor(gray.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            image[mi_y:ma_y, mi_x:ma_x] = cropped

            labels = [
                'bird',
                'boar',
                'dog',
                'dragon',
                'hare',
                'horse',
                'monkey',
                'ox',
                'ram',
                'rat',
                'serpent',
                'tiger'
            ]
            x = cropped
            x = cv2.resize(x, (224,224))
            x = tf.image.convert_image_dtype(x, tf.dtypes.uint8)
            x = tf.expand_dims(x, 0)
            pred = model.predict(x)[0]

            confidences = {}

            for label_image in label_images:
                confidences.update({label_image: 0})

            for i, p in enumerate(pred):
                confidences[labels[i]] = int(p*100)

    else:
        confidences = {}
        for label_image in label_images:
            confidences.update({label_image: 0})

    colors = {}
    for confidence in confidences.items():
        if confidence[1] > 50:
            colors.update({confidence[0]: (0, 0, 255)})
        else:
            colors.update({confidence[0]: (0, 0, 0)})

    h, w, c = image.shape

    b_shift = int(h//6)
    b_offset = 30

    image[b_shift*0:b_shift*1, 0:b_shift*1] = cv2.resize(label_images['bird'], (b_shift, b_shift)) 
    image[b_shift*1:b_shift*2, 0:b_shift*1] = cv2.resize(label_images['boar'], (b_shift, b_shift)) 
    image[b_shift*2:b_shift*3, 0:b_shift*1] = cv2.resize(label_images['dog'], (b_shift, b_shift)) 
    image[b_shift*3:b_shift*4, 0:b_shift*1] = cv2.resize(label_images['dragon'], (b_shift, b_shift)) 
    image[b_shift*4:b_shift*5, 0:b_shift*1] = cv2.resize(label_images['hare'], (b_shift, b_shift)) 
    image[b_shift*5:b_shift*6, 0:b_shift*1] = cv2.resize(label_images['horse'], (b_shift, b_shift)) 

    cv2.putText(image, str(confidences['bird']), (b_shift, b_shift*1-b_offset), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=colors['bird'], thickness=3)
    cv2.putText(image, str(confidences['boar']), (b_shift, b_shift*2-b_offset), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=colors['boar'], thickness=3)
    cv2.putText(image, str(confidences['dog']), (b_shift, b_shift*3-b_offset), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=colors['dog'], thickness=3)
    cv2.putText(image, str(confidences['dragon']), (b_shift, b_shift*4-b_offset), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=colors['dragon'], thickness=3)
    cv2.putText(image, str(confidences['hare']), (b_shift, b_shift*5-b_offset), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=colors['hare'], thickness=3)
    cv2.putText(image, str(confidences['horse']), (b_shift, b_shift*6-b_offset), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=colors['horse'], thickness=3)

    image[b_shift*0:b_shift*1, w-b_shift:w] = cv2.resize(label_images['monkey'], (b_shift, b_shift)) 
    image[b_shift*1:b_shift*2, w-b_shift:w] = cv2.resize(label_images['ox'], (b_shift, b_shift)) 
    image[b_shift*2:b_shift*3, w-b_shift:w] = cv2.resize(label_images['ram'], (b_shift, b_shift)) 
    image[b_shift*3:b_shift*4, w-b_shift:w] = cv2.resize(label_images['rat'], (b_shift, b_shift)) 
    image[b_shift*4:b_shift*5, w-b_shift:w] = cv2.resize(label_images['serpent'], (b_shift, b_shift)) 
    image[b_shift*5:b_shift*6, w-b_shift:w] = cv2.resize(label_images['tiger'], (b_shift, b_shift)) 

    cv2.putText(image, str(confidences['monkey']), (w-(b_shift+b_offset*4), b_shift*1-b_offset), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=colors['monkey'], thickness=3)
    cv2.putText(image, str(confidences['ox']), (w-(b_shift+b_offset*4), b_shift*2-b_offset), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=colors['ox'], thickness=3)
    cv2.putText(image, str(confidences['ram']), (w-(b_shift+b_offset*4), b_shift*3-b_offset), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=colors['ram'], thickness=3)
    cv2.putText(image, str(confidences['rat']), (w-(b_shift+b_offset*4), b_shift*4-b_offset), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=colors['rat'], thickness=3)
    cv2.putText(image, str(confidences['serpent']), (w-(b_shift+b_offset*4), b_shift*5-b_offset), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=colors['serpent'], thickness=3)
    cv2.putText(image, str(confidences['tiger']), (w-(b_shift+b_offset*4), b_shift*6-b_offset), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=colors['tiger'], thickness=3)

    cv2.imshow("Output", image)
    cv2.waitKey(1)
