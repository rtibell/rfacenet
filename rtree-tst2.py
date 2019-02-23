#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.

from mvnc import mvncapi as mvnc
from rtree import index
import numpy
import cv2
import sys
import os

IMAGES_DIR = 'images/'

VALIDATED_IMAGES_DIR = 'validated_images/'
validated_image_filename = VALIDATED_IMAGES_DIR + 'valid.jpg'

GRAPH_FILENAME = "facenet_celeb_ncs.graph"

# name of the opencv window
CV_WINDOW_NAME = "FaceNet"

# the same face will return 0.0
# different faces return higher numbers
# this is NOT between 0.0 and 1.0
#FACE_MATCH_THRESHOLD = 1.2
FACE_MATCH_THRESHOLD = 0.5


# Run an inference on the passed image
# image_to_classify is the image on which an inference will be performed
#    upon successful return this image will be overlayed with boxes
#    and labels identifying the found objects within the image.
# ssd_mobilenet_graph is the Graph object from the NCAPI which will
#    be used to peform the inference.
def run_inference(image_to_classify, facenet_graph):

    # get a resized version of the image that is the dimensions
    # SSD Mobile net expects
    print("run_inference")
    #cv2.imshow("preprocessed", image_to_classify)
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
    print('Total Difference is: ' + str(total_diff))

    if (total_diff < FACE_MATCH_THRESHOLD):
        # the total difference between the two is under the threshold so
        # the faces match.
        return True

    # differences between faces was over the threshold above so
    # they didn't match.
    return False


# Test all files in a list for a match against a valided face and display each one.
# valid_output is inference result for the valid image
# validated image filename is the name of the valid image file
# graph is the ncsdk Graph object initialized with the facenet graph file
#   which we will run the inference on.
# input_image_filename_list is a list of image files to compare against the
#   valid face output.
def run_images(valid_output, validated_image_filename, graph, input_image_filename_list):
    idxprop = index.Property()
    idxprop.dimension = 12
    idxprop.dat_extension = 'data'
    idxprop.idx_extension = 'index'
    idx = index.Index('3d_index',properties=idxprop)
    idx_nr = 0

    cv2.namedWindow(CV_WINDOW_NAME)
    for input_image_file in input_image_filename_list :
        print("Processing file ", input_image_file)
        # read one of the images to run an inference on from the disk
        infer_image = cv2.imread(IMAGES_DIR + input_image_file)

        # run a single inference on the image and overwrite the
        # boxes and labels
        test_output = run_inference(infer_image, graph)
        test_output = test_output[0:12]
        
        # R-tree
        print("building R-tree")
        targList = []
        for x in test_output:
            targList.append(x + 1.0 - 0.1) # xmin
        for x in test_output:
            targList.append(x + 1.0 + 0.1) # xmax
        targTupl =  tuple(targList)
        print("targTuble=",targTupl)
        idx.insert(idx_nr, targTupl, obj=idx_nr)
        a=list(idx.nearest(targTupl))
        print("R-tree near: ", a)
        idx_nr = idx_nr + 1

        # Test the inference results of this image with the results
        # from the known valid face.
        if (face_match(valid_output, test_output)):
            print('PASS!  File ' + input_image_file + ' matches ' + validated_image_filename)
        else:
            print('FAIL!  File ' + input_image_file + ' does not match ' + validated_image_filename)

        #cv2.imshow(CV_WINDOW_NAME, infer_image)

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

    validated_image = cv2.imread(validated_image_filename)
    valid_output = run_inference(validated_image, graph)

    print("Getting jpg files")
    # get list of all the .jpg files in the image directory
    input_image_filename_list = os.listdir(IMAGES_DIR)
    input_image_filename_list = [i for i in input_image_filename_list if i.endswith('.jpg')]
    if (len(input_image_filename_list) < 1):
        # no images to show
        print('No .jpg files found')
        return 1
    run_images(valid_output, validated_image_filename, graph, input_image_filename_list)

    print("Cleanup")
    # Clean up the graph and the device
    graph.DeallocateGraph()
    device.CloseDevice()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
