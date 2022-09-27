import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

def extract_number(img_name):
    global read
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray, 1.1, 4)

    for (x,y,w,h) in nplate:
        a,b = (int(0.02*img.shape[0]), int(0.025*img.shape[1]))
        plate = img[y+a:y+h-a, x+b:x+w-b, :]
        kernal = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernal, iterations=1)
        plate = cv2.erode(plate, kernal, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh, plate = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)

        read = pytesseract.image_to_string(plate)
        read = ''.join(e for e in read if e.isalnum())
        stat = read[0:2]

        print(read)
        cv2.rectangle(img, (x, y), (x+y, y+h), (51, 51, 255), 2)
        cv2.putText(img, read, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, (0, 0, 255), 3)
        cv2.imshow('Plate', plate)
    cv2.imshow("Result", img)
    cv2.imwrite("Result.png", img)
    cv2.waitkey(0)

extract_number("images\Cars0.png")