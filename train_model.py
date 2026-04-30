import cv2
import os
import numpy as np
import pickle

dataset_path = "dataset"

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
current_id = 0

for person in os.listdir(dataset_path):
    path = os.path.join(dataset_path, person)

    if not os.path.isdir(path):
        continue

    label_map[current_id] = person

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(current_id)

    current_id += 1

recognizer.train(faces, np.array(labels))

# save model
recognizer.save("trainer.yml")

# save labels
with open("labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("Training completed ✅")