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
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 5)
#     edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
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
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 5)
#     edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
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

#     output_filename = file.filename.split(".")[0] + "_cartoon.png"
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
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 9)
#     edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
#     color = cv2.bilateralFilter(img, 184, 250, 250)
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

#     img_gray = cv2.imread(file_path, 0)
#     img_gray = cv2.medianBlur(img_gray, 5)
#     img_edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
#     img_color = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
#     img_cartoon = cv2.bitwise_and(img_color, img_color, mask=img_edges)

#     output_filename = file.filename
#     output_path = os.path.join(OUTPUT_FOLDER, output_filename)
#     cv2.imwrite(output_path, img_cartoon)

#     download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
#     cv2.imwrite(download_path, img_cartoon)

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


# def remove_background(image_path: str) -> bytes:
#     with open(image_path, "rb") as f:
#         img = rembg.remove(f.read())

#     return img


# def resize_image(image_path: str, size: tuple[int, int]) -> np.ndarray:
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, size)
#     return img


# def convert_to_cartoon(image_path: str) -> np.ndarray:
#     img_gray = cv2.imread(image_path, 0)
#     img_gray = cv2.medianBlur(img_gray, 5)
#     img_edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
#     img_color = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
#     img_cartoon = cv2.bitwise_and(img_color, img_color, mask=img_edges)

#     return img_cartoon


# def merge_head_body(head_path: str, body_path: str) -> np.ndarray:
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
# async def remove_background_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return {"error": f"Failed to write file: {str(e)}"}

#     try:
#         image = remove_background(file_path)
#     except Exception as e:
#         return {"error": f"Failed to remove background: {str(e)}"}

#     output_filename, _ = os.path.splitext(file.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         with open(output_path, "wb") as f:
#             f.write(image)
#         with open(download_path, "wb") as f:
#             f.write(image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}


# @app.post("/resize")
# async def resize_image_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return {"error": f"Failed to write file: {str(e)}"}

#     try:
#         image = resize_image(file_path, (1080, 1080))
#     except Exception as e:
#         return {"error": f"Failed to resize image: {str(e)}"}

#     output_filename, _ = os.path.splitext(file.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         cv2.imwrite(output_path, image)
#         cv2.imwrite(download_path, image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}


# @app.post("/cartoon")
# async def convert_to_cartoon_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return {"error": f"Failed to write file: {str(e)}"}

#     try:
#         image = convert_to_cartoon(file_path)
#     except Exception as e:
#         return {"error": f"Failed to convert to cartoon: {str(e)}"}

#     output_filename, _ = os.path.splitext(file.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         cv2.imwrite(output_path, image)
#         cv2.imwrite(download_path, image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}


# @app.post("/headbody")
# async def merge_head_body_endpoint(head: UploadFile = File(...), body: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     head_path = os.path.join(TEMP_FOLDER, head.filename)
#     body_path = os.path.join(TEMP_FOLDER, body.filename)

#     try:
#         with open(head_path, "wb") as f:
#             f.write(await head.read())
#         with open(body_path, "wb") as f:
#             f.write(await body.read())
#     except Exception as e:
#         return {"error": f"Failed to write files: {str(e)}"}

#     try:
#         merged_image = merge_head_body(head_path, body_path)
#     except Exception as e:
#         return {"error": f"Failed to merge head and body: {str(e)}"}

#     if merged_image is None:
#         return {"error": "No face detected in the head image."}

#     output_filename, _ = os.path.splitext(head.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         cv2.imwrite(output_path, merged_image)
#         cv2.imwrite(download_path, merged_image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}


# @app.get("/download/{file_path}")
# def download_file(file_path: str) -> FileResponse:
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


# def remove_background(image_path: str) -> bytes:
#     with open(image_path, "rb") as f:
#         img = rembg.remove(f.read())
    
   

#     return img


# def resize_image(image_path: str, size: tuple[int, int]) -> np.ndarray:
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, size)
#     return img


# def convert_to_cartoon(image_path: str) -> np.ndarray:
#     img_gray = cv2.imread(image_path, 0)
#     img_gray = cv2.medianBlur(img_gray, 5)
#     img_edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
#     img_color = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
#     img_cartoon = cv2.bitwise_and(img_color, img_color, mask=img_edges)

#     return img_cartoon


# # def merge_head_body(head_path: str, body_path: str) -> np.ndarray:
# #     # Load haarcascade and shape predictor
# #     face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# #     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# #     # Load head and body images
# #     head = cv2.imread(head_path)
# #     body = cv2.imread(body_path)

# #     # Detect faces in the head image
# #     gray_head = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
# #     faces = face_cascade.detectMultiScale(gray_head, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# #     if len(faces) == 0:
# #         return None

# #     (x, y, w, h) = faces[0]

# #     # Detect face landmarks
# #     shape = predictor(gray_head, dlib.rectangle(x, y, x + w, y + h))
# #     landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)], dtype=np.int32)

# #     # Remove head from body if it already exists
# #     if np.max(body) > 0:
# #         body[landmarks[:, 1], landmarks[:, 0]] = [0, 0, 0]

# #     # Resize head to match body size
# #     resized_head = cv2.resize(head, (w, h))

# #     # Merge head and body
# #     merged_image = body.copy()
# #     merged_image[y:y + h, x:x + w] = resized_head

# #     return merged_image
# # def merge_head_body(head_path: str, body_path: str) -> np.ndarray:
# #     # Load haarcascade and shape predictor
# #     face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# #     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# #     # Load head and body images
# #     head = cv2.imread(head_path)
# #     body = cv2.imread(body_path)

# #     # Detect faces in the head image
# #     gray_head = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
# #     faces = face_cascade.detectMultiScale(gray_head, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# #     if len(faces) == 0:
# #         return None

# #     (x, y, w, h) = faces[0]

# #     # Detect face landmarks
# #     shape = predictor(gray_head, dlib.rectangle(x, y, x + w, y + h))
# #     if shape.num_parts != 68:
# #         return None

# #     landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)], dtype=np.int32)

# #     # Get the center of the face region
# #     center = np.mean(landmarks[17:21], axis=0, dtype=np.int32)

# #     # Calculate the translation vector to align the head with the body
# #     body_height, body_width = body.shape[:2]
# #     head_height, head_width = h, w
# #     tx = center[0] - head_width // 2
# #     ty = body_height - (center[1] + head_height) + (head_height // 4)

# #     # Translate the head image
# #     M = np.float32([[1, 0, tx], [0, 1, ty]])
# #     translated_head = cv2.warpAffine(head, M, (body_width, body_height))

# #     # Remove head from body if it already exists
# #     if np.max(body) > 0:
# #         translated_head[landmarks[:, 1] + ty, landmarks[:, 0] + tx] = [0, 0, 0]

# #     # Merge head and body
# #     merged_image = body.copy()
# #     merged_image += translated_head

# #     return merged_image
# def merge_head_body(head_path: str, body_path: str) -> np.ndarray:
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
# async def remove_background_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return {"error": f"Failed to write file: {str(e)}"}

#     try:
#         image = remove_background(file_path)
#     except Exception as e:
#         return {"error": f"Failed to remove background: {str(e)}"}

#     output_filename, _ = os.path.splitext(file.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         with open(output_path, "wb") as f:
#             f.write(image)
#         with open(download_path, "wb") as f:
#             f.write(image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}


# @app.post("/resize")
# async def resize_image_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return {"error": f"Failed to write file: {str(e)}"}

#     try:
#         image = resize_image(file_path, (1080, 1080))
#     except Exception as e:
#         return {"error": f"Failed to resize image: {str(e)}"}

#     output_filename, _ = os.path.splitext(file.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         cv2.imwrite(output_path, image)
#         cv2.imwrite(download_path, image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}


# @app.post("/cartoon")
# async def convert_to_cartoon_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return {"error": f"Failed to write file: {str(e)}"}

#     try:
#         image = convert_to_cartoon(file_path)
#         image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
#         resized_image = cv2.resize(image_gray, (1080, 1080))  # Resize to 1080x1080
#     except Exception as e:
#         return {"error": f"Failed to convert to cartoon: {str(e)}"}

#     output_filename, _ = os.path.splitext(file.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         cv2.imwrite(output_path, resized_image)
#         cv2.imwrite(download_path, resized_image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}


# @app.post("/headbody")
# async def merge_head_body_endpoint(head: UploadFile = File(...), body: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     head_path = os.path.join(TEMP_FOLDER, head.filename)
#     body_path = os.path.join(TEMP_FOLDER, body.filename)

#     try:
#         with open(head_path, "wb") as f:
#             f.write(await head.read())
#         with open(body_path, "wb") as f:
#             f.write(await body.read())
#     except Exception as e:
#         return {"error": f"Failed to write files: {str(e)}"}

#     try:
#         merged_image = merge_head_body(head_path, body_path)
#         if merged_image is None:
#             return {"error": "No face found in the head image."}
#     except Exception as e:
#         return {"error": f"Failed to merge head and body: {str(e)}"}

#     output_filename = f"merged_{uuid.uuid4().hex}.png"
#     output_path = os.path.join(OUTPUT_FOLDER, output_filename)
#     download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
#     try:
#         cv2.imwrite(output_path, merged_image)
#         cv2.imwrite(download_path, merged_image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}"
#     return {"link": image_link}


# @app.get("/download/{file_name}")
# async def download_file(file_name: str) -> FileResponse:
#     file_path = os.path.join(DOWNLOAD_FOLDER, file_name)
#     return FileResponse(file_path)


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)





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


# def remove_background(image_path: str) -> bytes:
#     with open(image_path, "rb") as f:
#         img = rembg.remove(f.read())
    
   

#     return img


# def resize_image(image_path: str, size: tuple[int, int]) -> np.ndarray:
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, size)
#     return img


# def convert_to_cartoon(image_path: str) -> np.ndarray:
#     img_gray = cv2.imread(image_path, 0)
#     img_gray = cv2.medianBlur(img_gray, 5)
#     img_edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
#     img_color = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
#     img_cartoon = cv2.bitwise_and(img_color, img_color, mask=img_edges)

#     return img_cartoon



# # def merge_head_body(head_path: str, body_path: str) -> np.ndarray:
# #     # Load haarcascade and shape predictor
# #     face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# #     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# #     # Load head and body images
# #     head = cv2.imread(head_path)
# #     body = cv2.imread(body_path)

# #     # Detect faces in the head image
# #     gray_head = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
# #     faces = face_cascade.detectMultiScale(gray_head, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# #     if len(faces) == 0:
# #         return None

# #     (x, y, w, h) = faces[0]

# #     # Detect face landmarks
# #     shape = predictor(gray_head, dlib.rectangle(x, y, x + w, y + h))
# #     landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)], dtype=np.int32)

# #     # Remove head from body if it already exists
# #     if np.max(body) > 0:
# #         body[landmarks[:, 1], landmarks[:, 0]] = [0, 0, 0]

# #     # Resize head to match body size
# #     resized_head = cv2.resize(head, (w, h))

# #     # Merge head and body
# #     merged_image = body.copy()
# #     merged_image[y:y + h, x:x + w] = resized_head

# #     return merged_image

# # def merge_head_body(head_path: str, body_path: str) -> np.ndarray:
# #     # Load haarcascade and shape predictor
# #     face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# #     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# #     # Load head and body images
# #     head = cv2.imread(head_path)
# #     body = cv2.imread(body_path)

# #     # Detect faces in the head image
# #     gray_head = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
# #     faces = face_cascade.detectMultiScale(gray_head, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# #     if len(faces) == 0:
# #         return None

# #     (x, y, w, h) = faces[0]

# #     # Detect face landmarks
# #     shape = predictor(gray_head, dlib.rectangle(x, y, x + w, y + h))
# #     landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)], dtype=np.int32)

# #     # Remove head from body if it already exists
# #     if np.max(body) > 0:
# #         body[landmarks[:, 1], landmarks[:, 0]] = [0, 0, 0]

# #     # Resize head to match body size
# #     resized_head = cv2.resize(head, (w, h))

# #     # Merge head and body
# #     merged_image = body.copy()
# #     merged_image[y:y + h, x:x + w] = resized_head

# #     # Extend the head region downwards to remove the gap
# #     head_bottom = y + h
# #     body_height, body_width, _ = body.shape
# #     extended_region = merged_image[head_bottom:body_height, x:x + w]
# #     extended_region[extended_region == 0] = body[head_bottom:body_height, x:x + w][extended_region == 0]

# #     return merged_image

# @app.post("/removebackground")
# async def remove_background_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return {"error": f"Failed to write file: {str(e)}"}

#     try:
#         image = remove_background(file_path)
#     except Exception as e:
#         return {"error": f"Failed to remove background: {str(e)}"}

#     output_filename, _ = os.path.splitext(file.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         with open(output_path, "wb") as f:
#             f.write(image)
#         with open(download_path, "wb") as f:
#             f.write(image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}


# @app.post("/resize")
# async def resize_image_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return {"error": f"Failed to write file: {str(e)}"}

#     try:
#         image = resize_image(file_path, (1080, 1080))
#     except Exception as e:
#         return {"error": f"Failed to resize image: {str(e)}"}

#     output_filename, _ = os.path.splitext(file.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         cv2.imwrite(output_path, image)
#         cv2.imwrite(download_path, image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}


# @app.post("/cartoon")
# async def convert_to_cartoon_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return {"error": f"Failed to write file: {str(e)}"}

#     try:
#         image = convert_to_cartoon(file_path)
#         image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
#         resized_image = cv2.resize(image_gray, (1080, 1080))  # Resize to 1080x1080
#     except Exception as e:
#         return {"error": f"Failed to convert to cartoon: {str(e)}"}

#     output_filename, _ = os.path.splitext(file.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         cv2.imwrite(output_path, resized_image)
#         cv2.imwrite(download_path, resized_image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}


# @app.post("/headbody")
# async def merge_head_body_endpoint(head: UploadFile = File(...), body: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     head_path = os.path.join(TEMP_FOLDER, head.filename)
#     body_path = os.path.join(TEMP_FOLDER, body.filename)

#     try:
#         with open(head_path, "wb") as f:
#             f.write(await head.read())
#         with open(body_path, "wb") as f:
#             f.write(await body.read())
#     except Exception as e:
#         return {"error": f"Failed to write files: {str(e)}"}

#     try:
#         merged_image = merge_head_body(head_path, body_path)
#         if merged_image is None:
#             return {"error": "No face found in the head image."}
#     except Exception as e:
#         return {"error": f"Failed to merge head and body: {str(e)}"}

#     output_filename = f"merged_{uuid.uuid4().hex}.png"
#     output_path = os.path.join(OUTPUT_FOLDER, output_filename)
#     download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
#     try:
#         cv2.imwrite(output_path, merged_image)
#         cv2.imwrite(download_path, merged_image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}"
#     return {"link": image_link}


# @app.get("/download/{file_name}")
# async def download_file(file_name: str) -> FileResponse:
#     file_path = os.path.join(DOWNLOAD_FOLDER, file_name)
#     return FileResponse(file_path)


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

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

# # Initialize dlib's face detector and shape predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# def create_folders():
#     os.makedirs(TEMP_FOLDER, exist_ok=True)
#     os.makedirs(OUTPUT_FOLDER, exist_ok=True)
#     os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)


# def remove_background(image_path: str) -> bytes:
#     with open(image_path, "rb") as f:
#         img = rembg.remove(f.read())

#     return img


# def resize_image(image_path: str, size: tuple[int, int]) -> np.ndarray:
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, size)
#     return img


# def convert_to_cartoon(image_path: str) -> np.ndarray:
#     img_gray = cv2.imread(image_path, 0)
#     img_gray = cv2.medianBlur(img_gray, 5)
#     img_edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
#     img_color = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
#     img_cartoon = cv2.bitwise_and(img_color, img_color, mask=img_edges)

#     return img_cartoon


# def merge_head_body(head_path: str, body_path: str) -> np.ndarray:
#     head_image = cv2.imread(head_path)
#     body_image = cv2.imread(body_path)

#     # Convert images to grayscale
#     head_gray = cv2.cvtColor(head_image, cv2.COLOR_BGR2GRAY)
#     body_gray = cv2.cvtColor(body_image, cv2.COLOR_BGR2GRAY)

#     # Detect faces in head image
#     head_faces = detector(head_gray)
#     if len(head_faces) == 0:
#         return None

#     # Extract head region using dlib face landmarks
#     head_landmarks = predictor(head_gray, head_faces[0])
#     head_points = np.array([(head_landmarks.part(n).x, head_landmarks.part(n).y) for n in range(68)], np.int32)
#     head_mask = np.zeros_like(head_gray)
#     cv2.fillPoly(head_mask, [head_points], 255)
#     head_region = cv2.bitwise_and(head_image, head_image, mask=head_mask)

#     # Resize body image to match head region size
#     body_resized = cv2.resize(body_image, (head_region.shape[1], head_region.shape[0]))

#     # Create a mask for the body region
#     body_mask = np.ones_like(head_mask) * 255

#     # Remove the head region from the body mask
#     cv2.fillPoly(body_mask, [head_points], 0)

#     # Merge head region and body region
#     merged_image = cv2.bitwise_or(head_region, body_resized)

#     # Apply the body mask
#     merged_image = cv2.bitwise_and(merged_image, merged_image, mask=body_mask)

#     return merged_image


# @app.post("/removebackground")
# async def remove_background_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return {"error": f"Failed to write file: {str(e)}"}

#     try:
#         image = remove_background(file_path)
#     except Exception as e:
#         return {"error": f"Failed to remove background: {str(e)}"}

#     output_filename, _ = os.path.splitext(file.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         with open(output_path, "wb") as f:
#             f.write(image)
#         with open(download_path, "wb") as f:
#             f.write(image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}


# @app.post("/resize")
# async def resize_image_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return {"error": f"Failed to write file: {str(e)}"}

#     try:
#         image = resize_image(file_path, (1080, 1080))
#     except Exception as e:
#         return {"error": f"Failed to resize image: {str(e)}"}

#     output_filename, _ = os.path.splitext(file.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         cv2.imwrite(output_path, image)
#         cv2.imwrite(download_path, image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}


# @app.post("/cartoon")
# async def convert_to_cartoon_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return {"error": f"Failed to write file: {str(e)}"}

#     try:
#         image = convert_to_cartoon(file_path)
#         image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
#         resized_image = cv2.resize(image_gray, (1080, 1080))  # Resize to 1080x1080
#     except Exception as e:
#         return {"error": f"Failed to convert to cartoon: {str(e)}"}

#     output_filename, _ = os.path.splitext(file.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         cv2.imwrite(output_path, resized_image)
#         cv2.imwrite(download_path, resized_image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}


# @app.post("/headbody")
# async def merge_head_body_endpoint(head: UploadFile = File(...), body: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     head_path = os.path.join(TEMP_FOLDER, head.filename)
#     body_path = os.path.join(TEMP_FOLDER, body.filename)

#     try:
#         with open(head_path, "wb") as f:
#             f.write(await head.read())
#         with open(body_path, "wb") as f:
#             f.write(await body.read())
#     except Exception as e:
#         return {"error": f"Failed to write files: {str(e)}"}

#     try:
#         merged_image = merge_head_body(head_path, body_path)
#         if merged_image is None:
#             return {"error": "No face found in the head image."}
#     except Exception as e:
#         return {"error": f"Failed to merge head and body: {str(e)}"}

#     output_filename = f"merged_{uuid.uuid4().hex}.png"
#     output_path = os.path.join(OUTPUT_FOLDER, output_filename)
#     download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
#     try:
#         cv2.imwrite(output_path, merged_image)
#         cv2.imwrite(download_path, merged_image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}"
#     return {"link": image_link}


# @app.get("/download/{file_name}")
# async def download_file(file_name: str) -> FileResponse:
#     file_path = os.path.join(DOWNLOAD_FOLDER, file_name)
#     return FileResponse(file_path)


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)

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

# # Initialize face detector and shape predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# def create_folders():
#     os.makedirs(TEMP_FOLDER, exist_ok=True)
#     os.makedirs(OUTPUT_FOLDER, exist_ok=True)
#     os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)


# def remove_background(image_path: str) -> bytes:
#     with open(image_path, "rb") as f:
#         img = rembg.remove(f.read())
#     return img


# def resize_image(image_path: str, size: tuple[int, int]) -> np.ndarray:
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, size)
#     return img


# def convert_to_cartoon(image_path: str) -> np.ndarray:
#     img_gray = cv2.imread(image_path, 0)
#     img_gray = cv2.medianBlur(img_gray, 5)
#     img_edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
#     img_color = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
#     img_cartoon = cv2.bitwise_and(img_color, img_color, mask=img_edges)
#     return img_cartoon


# def merge_head_body(head_path: str, body_path: str) -> np.ndarray:
#     # Load images
#     head_img = cv2.imread(head_path)
#     body_img = cv2.imread(body_path)

#     # Detect faces in the head image
#     head_gray = cv2.cvtColor(head_img, cv2.COLOR_BGR2GRAY)
#     head_faces = detector(head_gray)

#     # Check if a face was detected
#     if len(head_faces) == 0:
#         return None

#     # Assume only one face in the head image
#     head_landmarks = predictor(head_gray, head_faces[0])

#     # Get the bounding box coordinates of the face
#     left = head_landmarks.part(0).x
#     top = head_landmarks.part(19).y
#     right = head_landmarks.part(16).x
#     bottom = head_landmarks.part(9).y

#     # Resize the body image to match the size of the head region
#     body_resized = cv2.resize(body_img, (right - left, bottom - top))

#     # Replace the head region with the resized body image
#     head_img[top:bottom, left:right] = body_resized

#     return head_img


# @app.post("/removebackground")
# async def remove_background_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return {"error": f"Failed to write file: {str(e)}"}

#     try:
#         image = remove_background(file_path)
#     except Exception as e:
#         return {"error": f"Failed to remove background: {str(e)}"}

#     output_filename, _ = os.path.splitext(file.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         with open(output_path, "wb") as f:
#             f.write(image)
#         with open(download_path, "wb") as f:
#             f.write(image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}


# @app.post("/resize")
# async def resize_image_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return {"error": f"Failed to write file: {str(e)}"}

#     try:
#         image = resize_image(file_path, (1080, 1080))
#     except Exception as e:
#         return {"error": f"Failed to resize image: {str(e)}"}

#     output_filename, _ = os.path.splitext(file.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         cv2.imwrite(output_path, image)
#         cv2.imwrite(download_path, image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}


# @app.post("/cartoon")
# async def convert_to_cartoon_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return {"error": f"Failed to write file: {str(e)}"}

#     try:
#         image = convert_to_cartoon(file_path)
#         image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
#         resized_image = cv2.resize(image_gray, (1080, 1080))  # Resize to 1080x1080
#     except Exception as e:
#         return {"error": f"Failed to convert to cartoon: {str(e)}"}

#     output_filename, _ = os.path.splitext(file.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         cv2.imwrite(output_path, resized_image)
#         cv2.imwrite(download_path, resized_image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}


# @app.post("/headbody")
# async def merge_head_body_endpoint(head: UploadFile = File(...), body: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     head_path = os.path.join(TEMP_FOLDER, head.filename)
#     body_path = os.path.join(TEMP_FOLDER, body.filename)

#     try:
#         with open(head_path, "wb") as f:
#             f.write(await head.read())
#         with open(body_path, "wb") as f:
#             f.write(await body.read())
#     except Exception as e:
#         return {"error": f"Failed to write files: {str(e)}"}

#     try:
#         merged_image = merge_head_body(head_path, body_path)
#         if merged_image is None:
#             return {"error": "No face found in the head image."}
#     except Exception as e:
#         return {"error": f"Failed to merge head and body: {str(e)}"}

#     output_filename = f"merged_{uuid.uuid4().hex}.png"
#     output_path = os.path.join(OUTPUT_FOLDER, output_filename)
#     download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
#     try:
#         cv2.imwrite(output_path, merged_image)
#         cv2.imwrite(download_path, merged_image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}"
#     return {"link": image_link}


# @app.get("/download/{file_name}")
# async def download_file(file_name: str) -> FileResponse:
#     file_path = os.path.join(DOWNLOAD_FOLDER, file_name)
#     return FileResponse(file_path)


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import FileResponse
# import rembg
# import cv2
# import numpy as np
# import os
# import uuid
# import dlib
# from typing import Optional


# app = FastAPI()

# TEMP_FOLDER = "temp"
# OUTPUT_FOLDER = "output"
# DOWNLOAD_FOLDER = "download"

# def create_folders():
#     os.makedirs(TEMP_FOLDER, exist_ok=True)
#     os.makedirs(OUTPUT_FOLDER, exist_ok=True)
#     os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# def remove_background(image_path: str) -> bytes:
#     with open(image_path, "rb") as f:
#         img = rembg.remove(f.read())
#     return img

# def resize_image(image_path: str, size: tuple[int, int]) -> np.ndarray:
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, size)
#     return img

# def convert_to_cartoon(image_path: str) -> np.ndarray:
#     img_gray = cv2.imread(image_path, 0)
#     img_gray = cv2.medianBlur(img_gray, 5)
#     img_edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
#     img_color = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
#     img_cartoon = cv2.bitwise_and(img_color, img_color, mask=img_edges)
#     return img_cartoon

# def merge_head_body(head_path: str, body_path: str) -> np.ndarray:
#     detector = dlib.get_frontal_face_detector()

#     # Load head and body images
#     head_image = cv2.imread(head_path)
#     body_image = cv2.imread(body_path)

#     # Convert head and body images to grayscale
#     head_gray = cv2.cvtColor(head_image, cv2.COLOR_BGR2GRAY)
#     body_gray = cv2.cvtColor(body_image, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the head image
#     head_faces = detector(head_gray)

#     # Check if any faces were found
#     if len(head_faces) == 0:
#         return None

#     # Get the first detected face in the head image
#     head_face = head_faces[0]
#     head_x, head_y, head_w, head_h = head_face.left(), head_face.top(), head_face.width(), head_face.height()

#     # Calculate the center of the head face
#     head_center_x = head_x + head_w // 2
#     head_center_y = head_y + head_h // 2

#     # Resize the body image to match the size of the head face
#     resized_body = cv2.resize(body_image, (head_w, head_h))

#     # Calculate the region of interest (ROI) in the body image to replace with the head image
#     body_roi = body_gray[head_center_y - head_h // 2:head_center_y + head_h // 2,
#                head_center_x - head_w // 2:head_center_x + head_w // 2]

#     # Merge the head and body images
#     merged_image = np.copy(resized_body)
#     merged_image[body_roi > 0] = head_image[head_y:head_y + head_h, head_x:head_x + head_w][body_roi > 0]

#     return merged_image

# # def merge_head_body(head_path: str, body_path: str) -> Optional[np.ndarray]:
# #     head_image = cv2.imread(head_path)
# #     body_image = cv2.imread(body_path)

# #     if head_image is None or body_image is None:
# #         return None

# #     # Convert head image to grayscale
# #     head_gray = cv2.cvtColor(head_image, cv2.COLOR_BGR2GRAY)

# #     # Load face detector from dlib
# #     detector = dlib.get_frontal_face_detector()

# #     # Detect faces in the head image
# #     faces = detector(head_gray)

# #     if len(faces) == 0:
# #         return None

# #     # Extract the first detected face as the region of interest
# #     face = faces[0]

# #     # Get the coordinates of the face bounding box
# #     face_left = face.left()
# #     face_top = face.top()
# #     face_right = face.right()
# #     face_bottom = face.bottom()

# #     # Calculate the width and height of the face bounding box
# #     face_width = face_right - face_left
# #     face_height = face_bottom - face_top

# #     # Resize the body image to match the size of the face
# #     resized_body = cv2.resize(body_image, (face_width, face_height))

# #     # Replace the corresponding region in the body image with the head image
# #     merged_image = body_image.copy()
# #     merged_image[face_top:face_bottom, face_left:face_right] = resized_body

# #     return merged_image





# @app.post("/removebackground")
# async def remove_background_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return {"error": f"Failed to write file: {str(e)}"}

#     try:
#         image = remove_background(file_path)
#     except Exception as e:
#         return {"error": f"Failed to remove background: {str(e)}"}

#     output_filename, _ = os.path.splitext(file.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         with open(output_path, "wb") as f:
#             f.write(image)
#         with open(download_path, "wb") as f:
#             f.write(image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}

# @app.post("/resize")
# async def resize_image_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return {"error": f"Failed to write file: {str(e)}"}

#     try:
#         image = resize_image(file_path, (1080, 1080))
#     except Exception as e:
#         return {"error": f"Failed to resize image: {str(e)}"}

#     output_filename, _ = os.path.splitext(file.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         cv2.imwrite(output_path, image)
#         cv2.imwrite(download_path, image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}

# @app.post("/cartoon")
# async def convert_to_cartoon_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     file_path = os.path.join(TEMP_FOLDER, file.filename)
#     try:
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#     except Exception as e:
#         return {"error": f"Failed to write file: {str(e)}"}

#     try:
#         image = convert_to_cartoon(file_path)
#         image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
#         resized_image = cv2.resize(image_gray, (1080, 1080))  # Resize to 1080x1080
#     except Exception as e:
#         return {"error": f"Failed to convert to cartoon: {str(e)}"}

#     output_filename, _ = os.path.splitext(file.filename)
#     output_path = os.path.join(OUTPUT_FOLDER, f"{output_filename}.png")
#     download_path = os.path.join(DOWNLOAD_FOLDER, f"{output_filename}.png")
#     try:
#         cv2.imwrite(output_path, resized_image)
#         cv2.imwrite(download_path, resized_image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}.png"
#     return {"link": image_link}

# @app.post("/headbody")
# async def merge_head_body_endpoint(head: UploadFile = File(...), body: UploadFile = File(...)) -> dict[str, str]:
#     create_folders()

#     head_path = os.path.join(TEMP_FOLDER, head.filename)
#     body_path = os.path.join(TEMP_FOLDER, body.filename)

#     try:
#         with open(head_path, "wb") as f:
#             f.write(await head.read())
#         with open(body_path, "wb") as f:
#             f.write(await body.read())
#     except Exception as e:
#         return {"error": f"Failed to write files: {str(e)}"}

#     try:
#         merged_image = merge_head_body(head_path, body_path)
#         if merged_image is None:
#             return {"error": "No face found in the head image."}
#     except Exception as e:
#         return {"error": f"Failed to merge head and body: {str(e)}"}

#     output_filename = f"merged_{uuid.uuid4().hex}.png"
#     output_path = os.path.join(OUTPUT_FOLDER, output_filename)
#     download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
#     try:
#         cv2.imwrite(output_path, merged_image)
#         cv2.imwrite(download_path, merged_image)
#     except Exception as e:
#         return {"error": f"Failed to save image: {str(e)}"}

#     image_link = f"http://localhost:8000/download/{output_filename}"
#     return {"link": image_link}

# @app.get("/download/{file_name}")
# async def download_file(file_name: str) -> FileResponse:
#     file_path = os.path.join(DOWNLOAD_FOLDER, file_name)
#     return FileResponse(file_path)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# # from fastapi import FastAPI, Request, File, UploadFile
# # from fastapi.staticfiles import StaticFiles
# # from fastapi.templating import Jinja2Templates
# # from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
# # from werkzeug.utils import secure_filename
# # import cv2
# # from PIL import Image
# # import os
# # import dlib

# # app = FastAPI()
# # app.mount("/static", StaticFiles(directory="static"), name="static")
# # templates = Jinja2Templates(directory="templates")

# # TEMP_FOLDER = 'temp'
# # os.makedirs(TEMP_FOLDER, exist_ok=True)

# # @app.post("/headbody")
# # async def upload_files(request: Request, head: UploadFile = File(...), body: UploadFile = File(...)):
# #     head_path = os.path.join(TEMP_FOLDER, head.filename)
# #     body_path = os.path.join(TEMP_FOLDER, body.filename)

# #     with open(head_path, "wb") as head_image:
# #         head_image.write(await head.read())

# #     with open(body_path, "wb") as body_image:
# #         body_image.write(await body.read())

# #     # Process the files
# #     head_processed_file_path = os.path.join(TEMP_FOLDER, f"processed_{head.filename}")
# #     body_processed_file_path = os.path.join(TEMP_FOLDER, f"processed_{body.filename}")

# #     head_image = Image.open(head_path)
# #     head_image = head_image.convert("RGB")
# #     head_image.save(head_processed_file_path, format='JPEG')

# #     img = cv2.imread(body_path)
# #     detector = dlib.get_frontal_face_detector()
# #     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# #     faces = detector(img, 1)

# #     for i, face in enumerate(faces):
# #         landmarks = predictor(img, face)
# #         x = [landmarks.part(n).x for n in range(68)]
# #         y = [landmarks.part(n).y for n in range(68)]
# #         x1 = min(x) - int(0.2 * (max(x) - min(x)))
# #         y1 = min(y) - int(0.3 * (max(y) - min(y)))
# #         x2 = max(x) + int(0.2 * (max(x) - min(x)))
# #         y2 = max(y) + int(0.1 * (max(y) - min(y)))
# #         head_img = img[y1:y2, x1:x2]
# #         head_img = cv2.resize(head_img, (295, 294))
# #         gray = cv2.cvtColor(head_img, cv2.COLOR_BGR2GRAY)
# #         gray = cv2.medianBlur(gray, 5)
# #         edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
# #         color = cv2.bilateralFilter(head_img, 9, 250, 250)
# #         cartoon = cv2.bitwise_and(color, color, mask=edges)
# #         cv2.imwrite(body_processed_file_path, cartoon)

# #     output_filename = f"output_{head.filename}"
# #     output_file_path = os.path.join(TEMP_FOLDER, output_filename)

# #     if 'User-Agent' in request.headers and 'Mozilla' in request.headers['User-Agent']:
# #         return FileResponse(path=output_file_path, filename=output_filename)

# #     image_link = f"http://localhost:8000/download/{output_filename}"
# #     return JSONResponse(content={"image_link": image_link})

# # @app.get("/join_head/{head_filename}/{body_filename}")
# # def join_the_head(request: Request, head_filename: str, body_filename: str):
# #     head_file_path = os.path.join(TEMP_FOLDER, head_filename)
# #     body_file_path = os.path.join(TEMP_FOLDER, body_filename)

# #     head_image = cv2.imread(head_file_path)
# #     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# #     gray_image = cv2.cvtColor(head_image, cv2.COLOR_BGR2GRAY)
# #     faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

# #     if len(faces) == 0:
# #         return "No faces found in the head image."

# #     x, y, w, h = faces[0]
# #     face = head_image[y:y + h, x:x + w]

# #     body_image = cv2.imread(body_file_path)

# #     if body_image is None:
# #         return "Unable to read the body image."

# #     result = body_image.copy()

# #     body_height, body_width, _ = body_image.shape
# #     face_height, face_width, _ = face.shape

# #     if face_width > body_width:
# #         face = cv2.resize(face, (body_width - 40, (body_width - 40) * face_height // face_width))
# #         face_height, face_width, _ = face.shape
# #     elif face_height > body_height:
# #         face = cv2.resize(face, ((body_height - 230) * face_width // face_height, body_height - 230))
# #         face_height, face_width, _ = face.shape

# #     offset_x = 20
# #     offset_y = -250  # Adjust the offset_y value as needed
# #     start_x = (body_width - face.shape[1]) // 2 + offset_x
# #     start_y = (body_height - face.shape[0]) // 2 + offset_y

# #     if start_x >= 0 and start_y >= 0 and start_x + face_width <= body_width and start_y + face_height <= body_height:
# #         roi = result[start_y: start_y + face_height, start_x: start_x + face_width]
# #         face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
# #         _, mask = cv2.threshold(face_gray, 10, 255, cv2.THRESH_BINARY)
# #         mask_inv = cv2.bitwise_not(mask)
# #         result_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
# #         face_fg = cv2.bitwise_and(face, face, mask=mask)
# #         result[start_y: start_y + face_height, start_x: start_x + face_width] = cv2.add(result_bg, face_fg)

# #         output_filename = f"output_{head_filename}"
# #         output_file_path = os.path.join(TEMP_FOLDER, output_filename)
# #         cv2.imwrite(output_file_path, result)

# #         if 'User-Agent' in request.headers and 'Mozilla' in request.headers['User-Agent']:
# #             return RedirectResponse(url=f"/download/{output_filename}")
# #         else:
# #             image_link = f"http://localhost:8000/download/{output_filename}"
# #             return JSONResponse(content={"image_link": image_link})

# #     return "Error joining the head and body."

# # @app.get('/download/{filename}')
# # def download_file(filename: str):
# #     file_path = os.path.join(TEMP_FOLDER, filename)
# #     return FileResponse(path=file_path, filename=filename)

# # if __name__ == '__main__':
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8000)
###############################################################


###############################################################



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

def merge_head_body(head_path: str, body_path: str) -> np.ndarray:
    detector = dlib.get_frontal_face_detector()

    # Load head and body images
    head_image = cv2.imread(head_path)
    body_image = cv2.imread(body_path)

    # Convert head and body images to grayscale
    head_gray = cv2.cvtColor(head_image, cv2.COLOR_BGR2GRAY)
    body_gray = cv2.cvtColor(body_image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the head image
    head_faces = detector(head_gray)

    # Check if any faces were found
    if len(head_faces) == 0:
        return None

    # Get the first detected face in the head image
    head_face = head_faces[0]
    head_x, head_y, head_w, head_h = head_face.left(), head_face.top(), head_face.width(), head_face.height()

    # Calculate the center of the head face
    head_center_x = head_x + head_w // 2
    head_center_y = head_y + head_h // 2

    # Resize the body image to match the size of the head face
    resized_body = cv2.resize(body_image, (head_w, head_h))

    # Calculate the region of interest (ROI) in the body image to replace with the head image
    body_roi = body_gray[head_center_y - head_h // 2:head_center_y + head_h // 2,
               head_center_x - head_w // 2:head_center_x + head_w // 2]

    # Merge the head and body images
    merged_image = np.copy(resized_body)
    merged_image[body_roi > 0] = head_image[head_y:head_y + head_h, head_x:head_x + head_w][body_roi > 0]

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

@app.post("/headbody")
async def merge_head_body_endpoint(head: UploadFile = File(...), body: UploadFile = File(...)) -> dict[str, str]:
    create_folders()

    head_path = os.path.join(TEMP_FOLDER, head.filename)
    body_path = os.path.join(TEMP_FOLDER, body.filename)

    try:
        with open(head_path, "wb") as f:
            f.write(await head.read())
        with open(body_path, "wb") as f:
            f.write(await body.read())
    except Exception as e:
        return {"error": f"Failed to write files: {str(e)}"}

    try:
        merged_image = merge_head_body(head_path, body_path)
        if merged_image is None:
            return {"error": "No face found in the head image."}
    except Exception as e:
        return {"error": f"Failed to merge head and body: {str(e)}"}

    output_filename = f"merged_{uuid.uuid4().hex}.png"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
    try:
        cv2.imwrite(output_path, merged_image)
        cv2.imwrite(download_path, merged_image)
    except Exception as e:
        return {"error": f"Failed to save image: {str(e)}"}

    image_link = f"http://localhost:8000/download/{output_filename}"
    return {"link": image_link}

@app.get("/download/{file_name}")
async def download_file(file_name: str) -> FileResponse:
    file_path = os.path.join(DOWNLOAD_FOLDER, file_name)
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
   