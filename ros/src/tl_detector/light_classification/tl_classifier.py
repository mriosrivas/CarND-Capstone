from styx_msgs.msg import TrafficLight

import numpy as np
import os
import sys
import tensorflow as tf
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

path = os.path.dirname(os.path.abspath(__file__))

class TLClassifier(object):
    def __init__(self):

        #Object Detection Imports
        #PATH_TO_OBJECT_DETECTION = '/home/student/Desktop/CarND-Capstone/ros/src/tl_detector/light_classification/tensorflow/models/research/'
        PATH_TO_OBJECT_DETECTION = path + '/tensorflow/models/research/'
        sys.path.insert(0, PATH_TO_OBJECT_DETECTION)

        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as vis_util


        #Model Preparation
        #ssd_inception_sim_model = '/home/student/Desktop/CarND-Capstone/ros/src/tl_detector/light_classification/tensorflow/models/research/frozen_models/frozen_sim_inception/frozen_inference_graph.pb'
        ssd_inception_sim_model = path + '/tensorflow/models/research/frozen_models/frozen_sim_inception/frozen_inference_graph.pb'

        #ssd_inception_real_model = 'frozen_models/frozen_real_inception_6561/frozen_inference_graph.pb'

        #PATH_TO_LABELS = '/home/student/Desktop/CarND-Capstone/ros/src/tl_detector/light_classification/tensorflow/models/research/label_map.pbtxt'
        PATH_TO_LABELS = path + '/tensorflow/models/research/label_map.pbtxt'


        NUM_CLASSES = 14

        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)


        print("Success ")


    #def setModel(self, model):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(ssd_inception_sim_model, 'rb') as fid:
                serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')
        print("Success 2")


    #def startSession(self):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as self.sess:
                # Definite input and output Tensors for detection_graph
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                
                # Each box represents a part of the image where a particular object was detected.
                self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        print("Success 3")


    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        print("Success 4")



    def get_classification(self, img):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        #return TrafficLight.UNKNOWN
        #image = Image.open(img)
        image = Image.fromarray(img)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = self.load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        time0 = time.time()

        print("Before sess.run \n")

        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
          [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
          feed_dict={self.image_tensor: image_np_expanded})

        time1 = time.time()
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        #return scores, classes
        if scores[0]> 0.80: 
            if classes[0] and classes[1] and classes[2] == 1:
                print("GREEN Light", scores[0])
                return TrafficLight.GREEN
            elif classes[0] and classes[1] and classes[2] == 2:
                print("RED Light", scores[0])
                return TrafficLight.RED
            elif classes[0] and classes[1] and classes[2] == 3:
                print("YELLOW Light", scores[0])
                return TrafficLight.YELLOW
            else:
                print("UNKNOWN Light", scores[0])
                return TrafficLight.UNKNOWN

        else:
            print("UNKNOWN Light", scores[0])
            return TrafficLight.UNKNOWN


