import json 
import cv2 
import numpy as np 
import tensorflow as tf 

with open("apple-model/metadata.json") as f: 
    LABELS = json.load(f)["labels"]

model = tf.keras.models.load_model("keras_model.h5", compile=False)

cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    raise SystemExit("Could not open lol son")

print("webcam open, hit q to leave")
while True:
    ok, frame = cap.read()
    if not ok: 
        break 

    h,w = frame.shape[:2]
    s = min(h,w)
    crop = frame[(h - s) // 2:(h+s) // 2, (w + s) // 2]
    imh = cv2.resize(crop, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = img.astype(np.float32) / 127.5 - 1.0 

    probs = model.predict(x[None], verbose=0)[0]
    best = int(np.argmax(probs))
    text = f"{LABELS[best]}: {probs[best] * 100:.1f}%"

    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if best == 0 else (0, 0, 255), 3)
    for i, (label, p) in enumerate(zip(LABELS, probs)):
        cv2.putText(frame, f"{label}: {p * 100:.1f}%", (10, 80 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Apple Camera", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()
