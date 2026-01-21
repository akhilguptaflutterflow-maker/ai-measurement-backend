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
@app.post("/measure-napkin")
async def measure_napkin(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {"status": "ERROR", "message": "Invalid image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {"status": "ERROR", "message": "No object detected"}

    # Largest contour = object
    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    # ⚠️ Approx scale (tuning later)
    PIXEL_TO_CM = 0.12  # approx value

    width_cm = round(w * PIXEL_TO_CM, 2)
    height_cm = round(h * PIXEL_TO_CM, 2)

    return {
        "status": "OK",
        "object": "napkin_or_card",
        "width_cm": width_cm,
        "height_cm": height_cm
    }

    if brightness > 220:
        return {"status": "RETAKE", "reason": "Too bright"}

    return {"status": "OK"}
@app.post("/calibrate")
async def calibrate(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # A4 paper size in cm
    REAL_WIDTH_CM = 21.0
    REAL_HEIGHT_CM = 29.7

    pixel_to_cm_w = REAL_WIDTH_CM / w
    pixel_to_cm_h = REAL_HEIGHT_CM / h

    pixel_to_cm = round((pixel_to_cm_w + pixel_to_cm_h) / 2, 4)

    return {
        "status": "OK",
        "pixel_to_cm": pixel_to_cm
    }

