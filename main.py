#!/usr/bin/env python3

import time
import torch
import torchvision
import numpy as np
from torchvision import models, transforms
import torchvision.models.detection
import RPi.GPIO as GPIO
import cv2
from PIL import Image
from UltrasonicModule import getSonar
import time
from pygame import mixer

# Output pins for Raspi
buttonPin = 12
greenLed = 22
trigPin = 16
echoPin = 18

"""
# Creates an empty Image with the same dimensions as the taken image
emptyImage = torch.zeros([1080, 1920], dtype = torch.int16).numpy() 

# Creates filled rectangles for the left, middle and right area
leftArea = cv2.rectangle(emptyImage, (0,0), (640, 1080), (1), -1)
emptyImage = torch.zeros([1080, 1920], dtype = torch.int16).numpy() 

middleArea = cv2.rectangle(emptyImage, (640,0), (1280, 1080), (1), -1)
emptyImage = torch.zeros([1080, 1920], dtype = torch.int16).numpy() 

rightArea = cv2.rectangle(emptyImage, (1280,0), (1920, 1080), (1), -1)
emptyImage = torch.zeros([1080, 1920], dtype = torch.int16).numpy() 
"""
# Creates an empty Image with the same dimensions as the taken image
emptyImage = torch.zeros([270, 480], dtype = torch.int16).numpy() 

# Creates filled rectangles for the left, middle and right area
leftArea = cv2.rectangle(emptyImage, (0,0), (160, 270), (1), -1)
emptyImage = torch.zeros([270, 480], dtype = torch.int16).numpy() 

middleArea = cv2.rectangle(emptyImage, (160,0), (320, 270), (1), -1)
emptyImage = torch.zeros([270, 480], dtype = torch.int16).numpy() 

rightArea = cv2.rectangle(emptyImage, (320,0), (480, 270), (1), -1)
emptyImage = torch.zeros([270, 480], dtype = torch.int16).numpy()


# Capture Image from external Raspi camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)

# Setup for all external GPIO Pins
def setup():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(buttonPin, GPIO.IN, pull_up_down = GPIO.PUD_UP)
    GPIO.setup(greenLed, GPIO.OUT)
    GPIO.setup(trigPin, GPIO.OUT)   # set trigPin to OUTPUT mode
    GPIO.setup(echoPin, GPIO.IN)    # set echoPin to INPUT mode


# Function which is called after pressing the green button
def captureImage():
    global input_tensor
    global output
    global image
    image = cv2.imread('TestImage.jpg')
    image = cv2.resize(image, (480, 270))
    #ret, image = cap.read()
    #if not ret:
    #    raise RuntimeError("failed to read frame")
    input_tensor = np.moveaxis(image, -1, 0)
    input_tensor = torch.tensor(input_tensor)
    input_tensor = torch.unsqueeze(input_tensor, 0)/255

    # Load pretrained model
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights='DEFAULT')
    torch.save(model, '/home/pimci/Desktop/ObjectDetection_Fruits/new_untrainedmodel.pth')
    #model = torch.load('/home/pimci/Desktop/ObjectDetection_Fruits/banaApple_model.pth', map_location = 'cpu')
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        print(output)

        

def processImage():
    # No calculations for backpropagation
    with torch.no_grad():
        #images = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images = np.moveaxis(image, -1, 0)
        # converts numpy array to pytorch tensor
        images = torch.tensor(images) * 255
        # converts number format to uint8
        images = images.type(torch.uint8).cpu()
        pred_score = list(output[0]['scores'].detach().cpu().numpy())
        
        # if list of pred_scores is equal to 0 no objects were detected
        if len(pred_score) == 0:
            return print("No relevant objects detected")
        # if the first object of the list has a lower score than 0.7 no relevant objects were detected
        elif pred_score[0] < 0.7:
            return print("No relevant objects detected")
        else:
            pred_t = [pred_score.index(x) for x in pred_score if x > 0.7][-1]
            pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(output[0]['boxes'].detach().cpu().numpy())]
            pred_boxes = pred_boxes[:pred_t + 1]
            pred_boxes = torch.Tensor(pred_boxes)
            pred_boxes = torch.reshape(pred_boxes,(-1,4))

            labels = list(output[0]['labels'])
            labels = labels[:pred_t+1]
            
            for i in range(len(labels)):
              if labels[i] == 2:
                labels[i] = 'apple'
              else:
                labels[i] = 'banana'

            locationDict = {}
            
            for i in range(len(labels)):
                # Get each coordinate from the list
                xmin = pred_boxes[i][0]
                ymin = pred_boxes[i][1]
                xmax = pred_boxes[i][2]
                ymax = pred_boxes[i][3]
            
                emptyImage = torch.zeros([270, 480], dtype = torch.int16).numpy() 
                rectangleArea = cv2.rectangle(emptyImage,(int(xmin), int(ymin)),(int(xmax),int(ymax)), (1), -1)
              
                # Calculate Intersection over Union (A metric to find the overlap of two bounding boxes)
                iou_left = iou(rectangleArea, leftArea)
                iou_middle = iou(rectangleArea, middleArea)
                iou_right = iou(rectangleArea, rightArea)
                
              
                    
                if iou_left > iou_middle and iou_left > iou_right:
                    locationDict[i] = {labels[i]:'left'}
                elif iou_right > iou_middle and iou_right > iou_left:
                    locationDict[i] = {labels[i]:'right'}
                else:
                    locationDict[i] = {labels[i]:'middle'}
            
            banana_list = []
            apple_list = []
            for i in range(len(labels)): 
                if 'banana' in locationDict[i].keys():
                    banana_list.append(list(locationDict[i].values())[0])
                elif 'apple' in locationDict[i].keys():
                    apple_list.append(list(locationDict[i].values())[0])
            print(locationDict)
            # Function call to locate fruits
            locateFruits(banana_list, apple_list)
            
              
            
            boundingboxes = torchvision.utils.draw_bounding_boxes(images, pred_boxes, width=3, colors = 'red', labels = labels, font_size = 10)
            boundingboxes = boundingboxes.type(torch.float32)
            boundingboxes = torch.div(boundingboxes, 255)
            torchvision.utils.save_image(boundingboxes, '/home/pimci/Desktop/ObjectDetection_Fruits/OutputImages/safedapplebanana.jpg')



# function to calculate intersection over union
def iou(rectangle, area):
    intersection = (rectangle & area).sum()
    union = (rectangle | area).sum()
    return intersection/union



# function to locate in which region the most fruits are
def locateFruits(banana_list, apple_list):
    # For the case that most of the bananas are left:
    if banana_list.count('left') >= banana_list.count('middle') and banana_list.count('left') >= banana_list.count('right'):
        fruit_play_sound('banana', 'left')
    # For the case that most of the bananas are right:        
    if banana_list.count('right') >= banana_list.count('middle') and banana_list.count('right') >= banana_list.count('left'):
        fruit_play_sound('banana', 'right')
    # For the case that most of the bananas are in the middle:
    if banana_list.count('middle') >= banana_list.count('right') and banana_list.count('middle') >= banana_list.count('left'):
        fruit_play_sound('banana', 'middle')
    # For the case that no banana was found
    if banana_list.count('left') == 0 and banana_list.count('middle') == 0 and banana_list.count('right') == 0:
        print('Voice sound: No Banana')
    
    if apple_list.count('left') >= apple_list.count('middle') and apple_list.count('left') >= apple_list.count('right'):
        fruit_play_sound('apple', 'left')
    if apple_list.count('right') >= apple_list.count('middle') and apple_list.count('right') >= apple_list.count('left'):
        fruit_play_sound('apple', 'right')
    if apple_list.count('middle') >= apple_list.count('right') and apple_list.count('middle') >= apple_list.count('left'):
        fruit_play_sound('apple', 'middle')
    if apple_list.count('left') == 0 and apple_list.count('middle') == 0 and apple_list.count('right') == 0:
        print('Voice sound: No Apple')


def fruit_play_sound(fruit, direction):
    # First sound 
    if fruit == 'banana':
        mixer.init()
        mixer.music.load('Audiofiles/banana.mp3')
        mixer.music.play()
        while mixer.music.get_busy():
            time.sleep(0.1)
    elif fruit == 'apple':
        mixer.init()
        mixer.music.load('Audiofiles/apple.mp3')
        mixer.music.play()
        while mixer.music.get_busy():
            time.sleep(0.1)
    # Second sound
    if direction == 'left':
        mixer.music.load('Audiofiles/left.mp3')
        mixer.music.play()
        while mixer.music.get_busy():
          time.sleep(0.1)
    elif direction == 'right':
        mixer.music.load('Audiofiles/right.mp3')
        mixer.music.play()
        while mixer.music.get_busy():
          time.sleep(0.1)
    elif direction == 'middle':
        mixer.music.load('Audiofiles/middle.mp3')
        mixer.music.play()
        while mixer.music.get_busy():
          time.sleep(0.1)
        
    

def distance_sound(distance):
    mixer.init()
    if distance < 50:
        mixer.music.load('Audiofiles/distance_50.mp3')
        mixer.music.play()
        while mixer.music.get_busy():
            time.sleep(0.1)
    elif distance < 100:
        mixer.music.load('Audiofiles/distance_100.mp3')
        mixer.music.play()
        while mixer.music.get_busy():
            time.sleep(0.1)
    elif distance < 200:
        mixer.music.load('Audiofiles/distance_200.mp3')
        mixer.music.play()
        while mixer.music.get_busy():
            time.sleep(0.1)
    

# Performed main loop
def loop():
    while True:
        WaitForImage = True
        setup()
        #GPIO.output(vibration, True)
        #time.sleep(3)
        #GPIO.output(vibration, False)
        while WaitForImage == True:
            if GPIO.input(buttonPin) == GPIO.LOW:
                distance = getSonar()
                print(distance)
                distance_sound(distance)
                captureImage()
                processImage()
                WaitForImage = False
                destroy()
                print("Image taken and processed")
            
def destroy():
    GPIO.cleanup()
    cap.release()
    
if __name__ =='__main__':
    try:
        loop()
    except KeyboardInterrupt:
        destroy()
