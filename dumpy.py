

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import rembg
import cv2
import numpy as np
import os
import uuid
import dlib
from typing import Optional


app = FastAPI()

TEMP_FOLDER = "temp"
OUTPUT_FOLDER = "output"
DOWNLOAD_FOLDER = "download"

def create_folders():
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

def remove_background(image_path: str) -> bytes:
    with open(image_path, "rb") as f:
        img = rembg.remove(f.read())
    return img

def resize_image(image_path: str, size: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    return img

def convert_to_cartoon(image_path: str) -> np.ndarray:
    img_gray = cv2.imread(image_path, 0)
    img_gray = cv2.medianBlur(img_gray, 5)
    img_edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    img_color = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img_cartoon = cv2.bitwise_and(img_color, img_color, mask=img_edges)
    return img_cartoon


def merge_head_body(head_path: str, body_path: str) -> Optional[np.ndarray]:
    head_image = cv2.imread(head_path)
    body_image = cv2.imread(body_path)

    if head_image is None or body_image is None:
        return None

    # Convert head image to grayscale
    head_gray = cv2.cvtColor(head_image, cv2.COLOR_BGR2GRAY)

    # Load face detector from dlib
    detector = dlib.get_frontal_face_detector()

    # Detect faces in the head image
    faces = detector(head_gray)

    if len(faces) == 0:
        return None

    # Extract the first detected face as the region of interest
    face = faces[0]

    # Get the coordinates of the face bounding box
    face_left = face.left()
    face_top = face.top()
    face_right = face.right()
    face_bottom = face.bottom()

    # Calculate the width and height of the face bounding box
    face_width = face_right - face_left
    face_height = face_bottom - face_top

    # Resize the body image to match the size of the face
    resized_body = cv2.resize(body_image, (face_width, face_height))

    # Replace the corresponding region in the body image with the head image
    merged_image = body_image.copy()
    merged_image[face_top:face_bottom, face_left:face_right] = resized_body

    return merged_image





@app.post("/removebackground")
async def remove_background_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
    create_folders()

    file_path = os.path.join(TEMP_FOLDER, file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        return {"error": f"Failed to write file: {str(e)}"}

    try:
        image = remove_background(file_path)
    except Exception as e:
        return {"error": f"Failed to remove background: {str(e)}"}

    output_filename, _ = os.path.splitext(file.filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
    download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
    try:
        with open(output_path, "wb") as f:
            f.write(image)
        with open(download_path, "wb") as f:
            f.write(image)
    except Exception as e:
        return {"error": f"Failed to save image: {str(e)}"}

    image_link = f"http://localhost:8000/download/{output_filename}.png"
    return {"link": image_link}

@app.post("/resize")
async def resize_image_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
    create_folders()

    file_path = os.path.join(TEMP_FOLDER, file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        return {"error": f"Failed to write file: {str(e)}"}

    try:
        image = resize_image(file_path, (1080, 1080))
    except Exception as e:
        return {"error": f"Failed to resize image: {str(e)}"}

    output_filename, _ = os.path.splitext(file.filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
    download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
    try:
        cv2.imwrite(output_path, image)
        cv2.imwrite(download_path, image)
    except Exception as e:
        return {"error": f"Failed to save image: {str(e)}"}

    image_link = f"http://localhost:8000/download/{output_filename}.png"
    return {"link": image_link}

@app.post("/cartoon")
async def convert_to_cartoon_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
    create_folders()

    file_path = os.path.join(TEMP_FOLDER, file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        return {"error": f"Failed to write file: {str(e)}"}

    try:
        image = convert_to_cartoon(file_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        resized_image = cv2.resize(image_gray, (1080, 1080))  # Resize to 1080x1080
    except Exception as e:
        return {"error": f"Failed to convert to cartoon: {str(e)}"}

    output_filename, _ = os.path.splitext(file.filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
    download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
    try:
        cv2.imwrite(output_path, resized_image)
        cv2.imwrite(download_path, resized_image)
    except Exception as e:
        return {"error": f"Failed to save image: {str(e)}"}

    image_link = f"http://localhost:8000/download/{output_filename}.png"
    return {"link": image_link}

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from werkzeug.utils import secure_filename
import cv2
from PIL import Image
import os
import dlib



@app.post("/headbody")
async def upload_files(request: Request, head: UploadFile = File(...), body: UploadFile = File(...)):
    head_path = os.path.join(TEMP_FOLDER, head.filename)
    body_path = os.path.join(TEMP_FOLDER, body.filename)

    with open(head_path, "wb") as head_image:
        head_image.write(await head.read())

    with open(body_path, "wb") as body_image:
        body_image.write(await body.read())

    # Process the files
    head_processed_file_path = os.path.join(TEMP_FOLDER, f"processed_{head.filename}")
    body_processed_file_path = os.path.join(TEMP_FOLDER, f"processed_{body.filename}")

    head_image = Image.open(head_path)
    head_image = head_image.convert("RGB")
    head_image.save(head_processed_file_path, format='JPEG')

    img = cv2.imread(body_path)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces = detector(img, 1)

    for i, face in enumerate(faces):
        landmarks = predictor(img, face)
        x = [landmarks.part(n).x for n in range(68)]
        y = [landmarks.part(n).y for n in range(68)]
        x1 = min(x) - int(0.2 * (max(x) - min(x)))
        y1 = min(y) - int(0.3 * (max(y) - min(y)))
        x2 = max(x) + int(0.2 * (max(x) - min(x)))
        y2 = max(y) + int(0.1 * (max(y) - min(y)))
        head_img = img[y1:y2, x1:x2]
        head_img = cv2.resize(head_img, (295, 294))
        gray = cv2.cvtColor(head_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(head_img, 9, 250, 250)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        cv2.imwrite(body_processed_file_path, cartoon)

    output_filename = f"output_{head.filename}"
    output_file_path = os.path.join(TEMP_FOLDER, output_filename)

    if 'User-Agent' in request.headers and 'Mozilla' in request.headers['User-Agent']:
        return FileResponse(path=output_file_path, filename=output_filename)

    image_link = f"http://localhost:8000/download/{output_filename}"
    return JSONResponse(content={"image_link": image_link})

@app.get("/join_head/{head_filename}/{body_filename}")
def join_the_head(request: Request, head_filename: str, body_filename: str):
    head_file_path = os.path.join(TEMP_FOLDER, head_filename)
    body_file_path = os.path.join(TEMP_FOLDER, body_filename)

    head_image = cv2.imread(head_file_path)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(head_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return "No faces found in the head image."

    x, y, w, h = faces[0]
    face = head_image[y:y + h, x:x + w]

    body_image = cv2.imread(body_file_path)

    if body_image is None:
        return "Unable to read the body image."

    result = body_image.copy()

    body_height, body_width, _ = body_image.shape
    face_height, face_width, _ = face.shape

    if face_width > body_width:
        face = cv2.resize(face, (body_width - 40, (body_width - 40) * face_height // face_width))
        face_height, face_width, _ = face.shape
    elif face_height > body_height:
        face = cv2.resize(face, ((body_height - 230) * face_width // face_height, body_height - 230))
        face_height, face_width, _ = face.shape

    offset_x = 20
    offset_y = -250  # Adjust the offset_y value as needed
    start_x = (body_width - face.shape[1]) // 2 + offset_x
    start_y = (body_height - face.shape[0]) // 2 + offset_y

    if start_x >= 0 and start_y >= 0 and start_x + face_width <= body_width and start_y + face_height <= body_height:
        roi = result[start_y: start_y + face_height, start_x: start_x + face_width]
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(face_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        result_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        face_fg = cv2.bitwise_and(face, face, mask=mask)
        result[start_y: start_y + face_height, start_x: start_x + face_width] = cv2.add(result_bg, face_fg)

        output_filename = f"output_{head_filename}"
        output_file_path = os.path.join(TEMP_FOLDER, output_filename)
        cv2.imwrite(output_file_path, result)

        if 'User-Agent' in request.headers and 'Mozilla' in request.headers['User-Agent']:
            return RedirectResponse(url=f"/download/{output_filename}")
        else:
            image_link = f"http://localhost:8000/download/{output_filename}"
            return JSONResponse(content={"image_link": image_link})

    return "Error joining the head and body."

@app.get('/download/{filename}')
def download_file(filename: str):
    file_path = os.path.join(TEMP_FOLDER, filename)
    return FileResponse(path=file_path, filename=filename)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
