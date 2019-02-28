#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.

from mvnc import mvncapi as mvnc
from rtree import index
import numpy
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

# the same face will return 0.0
# different faces return higher numbers
# this is NOT between 0.0 and 1.0
#FACE_MATCH_THRESHOLD = 1.2
FACE_MATCH_THRESHOLD = 0.3

def extrFromList(lst):
    zlst = list(zip(*lst))
    return ([ min(l) for l in zlst ] , [ max(l) for l in zlst ])

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
    #print("run_inference")
    #cv2.imshow("Image", image_to_classify)
    #time.sleep(1)
    #cv2.waitKey(0)
    resized_image = preprocess_image(image_to_classify)
    #cv2.imshow("preprocessed", resized_image)

    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    facenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    output, userobj = facenet_graph.GetResult()

    #print("Total results: " + str(len(output)))
    #print(output)
    #print(userobj)
    return output


# whiten an image
def whiten_image(source_image):
    source_mean = numpy.mean(source_image)
    source_standard_deviation = numpy.std(source_image)
    std_adjusted = numpy.maximum(source_standard_deviation, 1.0 / numpy.sqrt(source_image.size))
    whitened_image = numpy.multiply(numpy.subtract(source_image, source_mean), 1 / std_adjusted)
    return whitened_image

# create a preprocessed image from the source image that matches the
# network expectations and return it
def preprocess_image(src):
    # scale the image
    NETWORK_WIDTH = 160
    NETWORK_HEIGHT = 160
    preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

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
        this_diff = numpy.square(face1_output[output_index] - face2_output[output_index])
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


def toRMSTuple(valueList):
    targList = []
    for x in valueList:
        targList.append(x*BIAS_RMS*(1.0-DELTA_PCT_RMS) - DELTA_ABS_RMS) # xmin
    for x in valueList:
        targList.append(x*BIAS_RMS*(1.0+DELTA_PCT_RMS) + DELTA_ABS_RMS) # xmax
    targTupl =  tuple(targList)
    return targTupl

def toNORTuple(valueList):
    targList = []
    for x in valueList:
        targList.append(x*BIAS_NOR*(1.0-DELTA_PCT_NOR) - DELTA_ABS_NOR) # xmin
    for x in valueList:
        targList.append(x*BIAS_NOR*(1.0+DELTA_PCT_NOR) + DELTA_ABS_NOR) # xmax
    targTupl =  tuple(targList)
    return targTupl

def setupIndex():
    idxprop = index.Property()
    idxprop.dimension = DIMENSIONS
    idxprop.dat_extension = 'data'
    idxprop.idx_extension = 'index'
    idx = index.Index('3d_index',properties=idxprop)
    return idx

def build_index(idx, graph, input_image_filename_list):
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

    (vmin, vmax) = extrFromList(testList)
    for x in range(0, len(input_image_filename_list)):
        test_output_full = testList[x]
        input_image_file = input_image_filename_list[x]
        test_output = rmsList(test_output_full, 13)
        testTupl = toRMSTuple(test_output)
        idx.insert(x, testTupl, obj=input_image_file)
        
        test_output = norList(test_output_full, 13, vmin, vmax)
        print("test_output: ", test_output)
        testTupl = toNORTuple(test_output)
        #print("testTupl: ", testTupl)
        idx.insert(10000+x, testTupl, obj=input_image_file)

    #print(testList)
    return (testList, vmin, vmax)

def search_index(index, validTupl, validated_image_filename):
        nearHits = list(index.nearest(validTupl, objects=True))
        nearHitsId = [str(item.id) + " -- " + item.object for item in nearHits]
        for h in nearHitsId :
            print("R-tree near for: " + validated_image_filename + ' matches ' + h)

        intrHits = list(index.intersection(validTupl, objects=True))
        intrHitsId = [str(item.id) + " -- " + item.object for item in intrHits]
        for h in intrHitsId :
            print("R-tree intersect for: " + validated_image_filename + ' matches ' + h)

def run_validate_images(validated_image_list, graph, index, testOutputList, testOutputFileList, lmin, lmax):

    for validated_image_filename in validated_image_list :
        validated_image = cv2.imread(VALIDATED_IMAGES_DIR + validated_image_filename)
        valid_output_full = run_inference(validated_image, graph, validated_image_filename)
        valid_output = rmsList(valid_output_full, 13)
        search_index(index, toRMSTuple(valid_output), validated_image_filename)
        valid_output = norList(valid_output_full, 13, lmin, lmax)
        search_index(index, toNORTuple(valid_output), validated_image_filename)

        # Test the inference results of this image with the results
        # from the known valid face.
        idx_nr = 0
        for test_output_full in testOutputList :
            (fm, fmdiff) = face_match(valid_output_full, test_output_full)
            if (fm):
                print('PASS!  test ' + validated_image_filename + ' matches ' + testOutputFileList[idx_nr] + ' id=' + str(idx_nr) + ' diff=' + str(fmdiff))
            idx_nr = idx_nr + 1


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
    index = setupIndex()
    (testOutputList, vmin, vmax) = build_index(index, graph, input_image_filename_list)
    validate_image_filename_list = os.listdir(VALIDATED_IMAGES_DIR)
    run_validate_images(validate_image_filename_list, graph, index, testOutputList, input_image_filename_list, vmin, vmax)

    print("Cleanup")
    # Clean up the graph and the device
    graph.DeallocateGraph()
    device.CloseDevice()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
