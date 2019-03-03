#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.

from mvnc import mvncapi as mvnc
from scipy.spatial import KDTree
import numpy as np
import cv2
import sys
import os
import math
import time

IMAGES_DIR = '../../Face-DB/colorferet/images/jpg_train/'
IMAGES_DIR = '/home/pi/Projects/Movidius/Face-DB/colorferet/images/jpg_train/'
VALIDATED_IMAGES_DIR = '/home/pi/Projects/Movidius/Face-DB/colorferet/images/jpg_test/'
GRAPH_FILENAME = "facenet_celeb_ncs.graph"
DIMENSIONS=10
BIAS_RMS=100
DELTA_ABS_RMS=0.13
DELTA_PCT_RMS=0.04
BIAS_NOR=100
DELTA_ABS_NOR=0.25
DELTA_PCT_NOR=0.0090

# name of the opencv window
CV_WINDOW_NAME = "FaceNet"

# Create the haar cascade
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# the same face will return 0.0
# different faces return higher numbers
# this is NOT between 0.0 and 1.0
#FACE_MATCH_THRESHOLD = 1.2
FACE_MATCH_THRESHOLD = 0.3

#def extrFromList(lst):
#    zlst = list(zip(*lst))
#    return ([ min(l) for l in zlst ] , [ max(l) for l in zlst ])

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        offs = min(i + n, len(l))
        yield l[i:offs]

def rms(valList):
    """Calculates the RMS value of all values in list valList."""
    sum = 0
    for v in valList:
        sum = sum + v*v
    return math.sqrt(sum)/len(valList)

def rmsDiff(val1, val2):
    """Calculates the RMS value of all values in list valList."""
    sum = 0
    for i in range(0, len(val1)):
        v = val1[i] - val2[i]
        sum = sum + v*v
    return math.sqrt(sum)/len(val1)

def norm(valList, lmin, lmax):
    """Calculates the norm value between 0 and 1.0, of all values in list valList."""
    retList = []
    for i in range(0,len(valList)):
        x = (valList[i]-lmin[i])/lmax[i]
        retList.append(x)
    return retList

def rmsList(l, s):
    """Chops the list l into s partitions, calculates the RMS of sublist and returns a list of size s with RMS values."""
    chList = chunks(l, s)
    return [rms(x) for x in chList]

def norList(l, s, lmin, lmax):
    """Chops the list l into s partitions, calculates the RMS of sublist and returns a list of size s with RMS values."""
    chList = chunks(norm(l, lmin, lmax), s)
    return [rms(x) for x in chList]


# Run an inference on the passed image
# image_to_classify is the image on which an inference will be performed
#    upon successful return this image will be overlayed with boxes
#    and labels identifying the found objects within the image.
# ssd_mobilenet_graph is the Graph object from the NCAPI which will
#    be used to peform the inference.
def run_inference(image_to_classify, facenet_graph, file_name):
    # get a resized version of the image that is the dimensions
    # SSD Mobile net expects
    #if file_name.startswith("Donald") or file_name.startswith("Bara"):
    #    #cv2.imshow("Image", image_to_classify)
    #    #print("image",file_name)
    #    #cv2.waitKey(0)
    resized_image = preprocess_image(image_to_classify, file_name)
    #if file_name.startswith("Donald") or file_name.startswith("Bara"):
    #    cv2.imshow("Image rez", resized_image)
    #    print("image rez",file_name)
    #    #cv2.waitKey(0)

    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    facenet_graph.LoadTensor(resized_image.astype(np.float16), None)

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    output, userobj = facenet_graph.GetResult()
    return output

# Crop the face using opencv
def crop_face(image_to_classify, file_name, minWidth, minHeight):
    gray = cv2.cvtColor(image_to_classify, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(minWidth, minHeight)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )
    (height, width, channels) = image_to_classify.shape
    if (len(faces) < 1):
        print("Found no faces for {0}! W*H: {1}, {2}".format(file_name, width, height))
        return image_to_classify
    print("Found {0} faces for {1}! W*H: {2}, {3}".format(len(faces), file_name, width, height))
    (x, y, w, h) = faces[0]
    #print("Face at {0}, {1}, {2}, {3}".format(x,y,w,h))
    return image_to_classify[y:y+h, x:x+w]

# whiten an image
def whiten_image(source_image):
    source_mean = np.mean(source_image)
    source_standard_deviation = np.std(source_image)
    std_adjusted = np.maximum(source_standard_deviation, 1.0 / np.sqrt(source_image.size))
    whitened_image = np.multiply(np.subtract(source_image, source_mean), 1 / std_adjusted)
    return whitened_image

# create a preprocessed image from the source image that matches the
# network expectations and return it
def preprocess_image(src, file_name):
    NETWORK_WIDTH = 160
    NETWORK_HEIGHT = 160
    # detect face
    MIN_USE_WIDTH = 80
    MIN_USE_HEIGHT = 80
    img_src = crop_face(src, file_name,  MIN_USE_WIDTH, MIN_USE_HEIGHT) 

    # scale the image
    preprocessed_image = cv2.resize(img_src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    #convert to RGB
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)

    #whiten
    preprocessed_image = whiten_image(preprocessed_image)

    # return the preprocessed image
    return preprocessed_image


# determine if two images are of matching faces based on the
# the network output for both images.
def face_match(face1_output, face2_output):
    if (len(face1_output) != len(face2_output)):
        print('length mismatch in face_match')
        return False
    total_diff = 0
    for output_index in range(0, len(face1_output)):
        this_diff = np.square(face1_output[output_index] - face2_output[output_index])
        total_diff += this_diff
    #print('Total Difference is: ' + str(total_diff))

    if (total_diff < FACE_MATCH_THRESHOLD):
        # the total difference between the two is under the threshold so
        # the faces match.
        #print('Total Difference is: ' + str(total_diff))
        return (True, total_diff)

    # differences between faces was over the threshold above so
    # they didn't match.
    return (False, 0)


def build_index(graph, input_image_filename_list):
    idx_nr = 0
    testList = []
    cv2.namedWindow(CV_WINDOW_NAME)
    for input_image_file in input_image_filename_list :
        print("Processing [",idx_nr,"] file ", input_image_file)
        # read one of the images to run an inference on from the disk
        infer_image = cv2.imread(IMAGES_DIR + input_image_file)
        # run a single inference on the image and overwrite the
        # boxes and labels
        test_output_full = run_inference(infer_image, graph, input_image_file + " -- " + str(idx_nr))
        testList.append(test_output_full)
        idx_nr = idx_nr + 1
    points = np.array(testList)
    tree = KDTree(points)
    #print(testList)
    return ( tree, testList )

def search_index(tree, validTuple, validated_image_filename, testOutputList, fileNameList):
    (valAtIdx, idx) = tree.query(validTuple)
    #print(idx, valAtIdx)
    elem = testOutputList[idx] 
    print("KD-tree match val " + str(valAtIdx) + " rms " + str(rmsDiff(elem, validTuple)) + " images " + fileNameList[idx] + " -- " + validated_image_filename)


def run_validate_images(validated_image_list, graph, tree, testOutputList, testOutputFileList):
    for validated_image_filename in validated_image_list :
        validated_image = cv2.imread(VALIDATED_IMAGES_DIR + validated_image_filename)
        valid_output_full = run_inference(validated_image, graph, validated_image_filename)
        search_index(tree, valid_output_full, validated_image_filename, testOutputList, testOutputFileList)


# This function is called from the entry point to do
# all the work of the program
def main():

    # Get a list of ALL the sticks that are plugged in
    # we need at least one
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No NCS devices found')
        quit()

    # Pick the first stick to run the network
    device = mvnc.Device(devices[0])

    # Open the NCS
    device.OpenDevice()

    # The graph file that was created with the ncsdk compiler
    graph_file_name = GRAPH_FILENAME

    # read in the graph file to memory buffer
    print("Read graph into memory")
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()

    # create the NCAPI graph instance from the memory buffer containing the graph file.
    graph = device.AllocateGraph(graph_in_memory)
    print("Graph loaded into memory")

    print("Getting jpg files")
    # get list of all the .jpg files in the image directory
    input_image_filename_list = os.listdir(IMAGES_DIR)
    input_image_filename_list = [i for i in input_image_filename_list if i.endswith('.jpg')]
    if (len(input_image_filename_list) < 1):
        # no images to show
        print('No .jpg files found')
        return 1
    (tree, testOutputList) = build_index( graph, input_image_filename_list)
    validate_image_filename_list = os.listdir(VALIDATED_IMAGES_DIR)
    run_validate_images(validate_image_filename_list, graph, tree, testOutputList, input_image_filename_list)

    print("Cleanup")
    # Clean up the graph and the device
    graph.DeallocateGraph()
    device.CloseDevice()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
