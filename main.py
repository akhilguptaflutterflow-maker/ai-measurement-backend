from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np

app = FastAPI()

@app.post("/check-image")
async def check_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {"status": "RETAKE", "reason": "Invalid image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)

    if blur_score < 100:
        return {"status": "RETAKE", "reason": "Image blurry"}

    if brightness < 50:
        return {"status": "RETAKE", "reason": "Too dark"}

    if brightness > 220:
        return {"status": "RETAKE", "reason": "Too bright"}

    return {"status": "OK"}
