# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import FileResponse
# import rembg
# import cv2
# import numpy as np
# import os
# import uuid
# import dlib

# app = FastAPI()

# TEMP_FOLDER = "temp"
# OUTPUT_FOLDER = "output"
# DOWNLOAD_FOLDER = "download"


# def create_folders():
#     os.makedirs(TEMP_FOLDER, exist_ok=True)
#     os.makedirs(OUTPUT_FOLDER, exist_ok=True)
#     os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)


# def remove_background(image_path):
#     with open(image_path, "rb") as f:
#         img = rembg.remove(f.read())
        
#     return img


# def resize_image(image_path, size):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, size)
#     return img


# def convert_to_cartoon(image_path):
#     img = cv2.imread(image_path)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_gray = cv2.medianBlur(img_gray, 5)
#     edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
#     color = cv2.bilateralFilter(img, 9, 250, 250)
#     cartoon = cv2.bitwise_and(color, color, mask=edges)
#     return cartoon



# def merge_head_body(head_path, body_path):
#     # Load haarcascade and shape predictor
#     face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#     # Load head and body images
#     head = cv2.imread(head_path)
#     body = cv2.imread(body_path)

#     # Detect faces in the head image
#     gray_head = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray_head, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     if len(faces) == 0:
#         return None

#     (x, y, w, h) = faces[0]

#     # Detect face landmarks
#     shape = predictor(gray_head, dlib.rectangle(x, y, x + w, y + h))
#     landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)], dtype=np.int32)

#     # Remove head from body if it already exists
#     if np.max(body) > 0:
#         body[landmarks[:, 1], landmarks[:, 0]] = [0, 0, 0]

#     # Resize head to match body size
#     resized_head = cv2.resize(head, (w, h))

#     # Merge head and body
#     merged_image = body.copy()
#     merged_image[y:y + h, x:x + w] = resized_head

#     return merged_image


# @app.post("/removebackground")
# async def remove_background_endpoint(file: UploadFile = File(...)):
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     image = remove_background(file_path)

#     output_filename = file.filename
#     output_path = os.path.join(OUTPUT_FOLDER, output_filename)
#     with open(output_path, "wb") as f:
#         f.write(image)

#     download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
#     with open(download_path, "wb") as f:
#         f.write(image)

#     image_link = f"http://localhost:8000/download/{output_filename}"
#     return {"link": image_link}


# @app.post("/resize")
# async def resize_image_endpoint(file: UploadFile = File(...)):
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     image = resize_image(file_path, (1080, 1080))

#     output_filename = file.filename
#     output_path = os.path.join(OUTPUT_FOLDER, output_filename)
#     cv2.imwrite(output_path, image)

#     download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
#     cv2.imwrite(download_path, image)

#     image_link = f"http://localhost:8000/download/{output_filename}"
#     return {"link": image_link}



# @app.post("/cartoon")
# async def convert_to_cartoon_endpoint(file: UploadFile = File(...)):
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     image = convert_to_cartoon(file_path)

#     output_filename = file.filename
#     output_path = os.path.join(OUTPUT_FOLDER, output_filename)
#     cv2.imwrite(output_path, image)

#     download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
#     cv2.imwrite(download_path, image)

#     image_link = f"http://localhost:8000/download/{output_filename}"
#     return {"link": image_link}


# @app.post("/headbody")
# async def merge_head_body_endpoint(head: UploadFile = File(...), body: UploadFile = File(...)):
#     create_folders()

#     head_path = os.path.join(TEMP_FOLDER, head.filename)
#     with open(head_path, "wb") as f:
#         f.write(await head.read())

#     body_path = os.path.join(TEMP_FOLDER, body.filename)
#     with open(body_path, "wb") as f:
#         f.write(await body.read())

#     merged_image = merge_head_body(head_path, body_path)

#     if merged_image is None:
#         return {"error": "No face detected in the head image."}

#     output_filename = head.filename
#     output_path = os.path.join(OUTPUT_FOLDER, output_filename)
#     cv2.imwrite(output_path, merged_image)

#     download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
#     cv2.imwrite(download_path, merged_image)

#     image_link = f"http://localhost:8000/download/{output_filename}"
#     return {"link": image_link}


# @app.get("/download/{file_path}")
# def download_file(file_path: str):
#     image_path = os.path.join(DOWNLOAD_FOLDER, file_path)
#     return FileResponse(image_path, media_type="image/png")


# @app.on_event("startup")
# async def startup_event():
#     create_folders()


# @app.on_event("shutdown")
# async def shutdown_event():
#     # Clean up temporary and output directories
#     for file_name in os.listdir(TEMP_FOLDER):
#         file_path = os.path.join(TEMP_FOLDER, file_name)
#         os.remove(file_path)

#     for file_name in os.listdir(OUTPUT_FOLDER):
#         file_path = os.path.join(OUTPUT_FOLDER, file_name)
#         os.remove(file_path)
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import rembg
import cv2
import numpy as np
import os
import uuid

app = FastAPI()

TEMP_FOLDER = "temp"
OUTPUT_FOLDER = "output"
DOWNLOAD_FOLDER = "download"


def create_folders():
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)


def remove_background(image_path):
    with open(image_path, "rb") as f:
        img = rembg.remove(f.read())

    return img


def resize_image(image_path, size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    return img


def convert_to_cartoon(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 5)
    edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


def merge_head_body(head_path, body_path):
    # Load haarcascade
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Load head and body images
    head = cv2.imread(head_path)
    body = cv2.imread(body_path)

    # Convert head image to grayscale
    gray_head = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)

    # Detect faces in the head image
    faces = face_cascade.detectMultiScale(gray_head, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]

    # Resize head to match body size
    resized_head = cv2.resize(head, (w, h))

    # Merge head and body
    merged_image = body.copy()
    merged_image[y:y + h, x:x + w] = resized_head

    return merged_image


@app.post("/removebackground")
async def remove_background_endpoint(file: UploadFile = File(...)):
    create_folders()

    file_path = os.path.join(TEMP_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    image = remove_background(file_path)

    output_filename = file.filename
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    with open(output_path, "wb") as f:
        f.write(image)

    download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
    with open(download_path, "wb") as f:
        f.write(image)

    image_link = f"http://localhost:8000/download/{output_filename}"
    return {"link": image_link}


@app.post("/resize")
async def resize_image_endpoint(file: UploadFile = File(...)):
    create_folders()

    file_path = os.path.join(TEMP_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    image = resize_image(file_path, (1080, 1080))

    output_filename = file.filename
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, image)

    download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
    cv2.imwrite(download_path, image)

    image_link = f"http://localhost:8000/download/{output_filename}"
    return {"link": image_link}


@app.post("/cartoon")
async def convert_to_cartoon_endpoint(file: UploadFile = File(...)):
    create_folders()

    file_path = os.path.join(TEMP_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    image = convert_to_cartoon(file_path)

    output_filename = file.filename
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, image)

    download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
    cv2.imwrite(download_path, image)

    image_link = f"http://localhost:8000/download/{output_filename}"
    return {"link": image_link}


@app.post("/headbody")
async def merge_head_body_endpoint(head: UploadFile = File(...), body: UploadFile = File(...)):
    create_folders()

    head_path = os.path.join(TEMP_FOLDER, head.filename)
    with open(head_path, "wb") as f:
        f.write(await head.read())

    body_path = os.path.join(TEMP_FOLDER, body.filename)
    with open(body_path, "wb") as f:
        f.write(await body.read())

    merged_image = merge_head_body(head_path, body_path)

    if merged_image is None:
        return {"error": "No face detected in the head image."}

    output_filename = head.filename
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, merged_image)

    download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
    cv2.imwrite(download_path, merged_image)

    image_link = f"http://localhost:8000/download/{output_filename}"
    return {"link": image_link}


@app.get("/download/{file_path}")
def download_file(file_path: str):
    image_path = os.path.join(DOWNLOAD_FOLDER, file_path)
    return FileResponse(image_path, media_type="image/png")


@app.on_event("startup")
async def startup_event():
    create_folders()


@app.on_event("shutdown")
async def shutdown_event():
    # Clean up temporary and output directories
    for file_name in os.listdir(TEMP_FOLDER):
        file_path = os.path.join(TEMP_FOLDER, file_name)
        os.remove(file_path)

    for file_name in os.listdir(OUTPUT_FOLDER):
        file_path = os.path.join(OUTPUT_FOLDER, file_name)
        os.remove(file_path)
