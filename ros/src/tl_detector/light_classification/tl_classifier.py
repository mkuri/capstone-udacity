from styx_msgs.msg import TrafficLight

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

VALID_SCORE = 0.3
ID_CLASS_TL = 10

class TLClassifier(object):
    def __init__(self, model_name):
        #TODO load classifier
        model_path = './light_classification/'
        graph_path = model_path + model_name + '/frozen_inference_graph.pb'

        self.graph_detection = self.load_graph(graph_path)

        self.x = self.graph_detection.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.graph_detection.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.graph_detection.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.graph_detection.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.graph_detection.get_tensor_by_name('num_detections:0')


    def load_graph(self, graph_path):
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

        return graph


    def detect_light(self, cv_image):
        x = np.asarray(cv_image, dtype='uint8')
        x = np.expand_dims(x, axis=0)

        with tf.Session(graph=self.graph_detection) as sess:
            (boxes, scores, classes, num) = sess.run(
                    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                    feed_dict={self.x: x})

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        max_score = 0
        for i, classid in enumerate(classes):
            if (classid == ID_CLASS_TL) and (scores[i] > max_score):
                max_score = scores[i]
                idx_light = i

        if max_score < VALID_SCORE:
            return None

        box = boxes[idx_light]
        height = cv_image.shape[0]
        width = cv_image.shape[1]

        bbox = np.array([box[0]*height, box[1]*width, box[2]*height, box[3]*width]).astype(int)
        image_light = cv_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        plt.imshow(image_light)
        plt.show()

        return image_light

    def classify_light_state(self, image_light):
        hsv_img = cv2.cvtColor(image_light, cv2.COLOR_BGR2HSV)
        h = hsv_img[:, :, 0]
        s = hsv_img[:, :, 1]

        height = image_light.shape[0]
        width = image_light.shape[1]
        n_pixels = height * width
        threshould = n_pixels / 20

        plt.hist(h, bins=10)
        plt.show()

        plt.hist(s, bins=10)
        plt.show()

        red = np.zeros(h.shape, dtype=np.uint8)
        red[((h < 30/360*256) | (h > 300/360*256)) & (s > 100)] = 1
        n_red = np.count_nonzero(red)
        print('n_red:', n_red)

        yellow = np.zeros(h.shape, dtype=np.uint8)
        yellow[((40/360*256 < h) & (h < 80/360*256)) & (s > 100)] = 1
        n_yellow = np.count_nonzero(yellow)
        print('n_yellow:', n_yellow)

        green = np.zeros(h.shape, dtype=np.uint8)
        green[((120/360*256 < h) & (h < 260/360*256)) & (s > 100)] = 1
        n_green = np.count_nonzero(green)
        print('n_green:', n_green)

        if (n_red > n_yellow) and (n_red > n_green) and (n_red > threshould):
            return TrafficLight.RED
        elif (n_yellow > n_green) and (n_yellow > threshould):
            return TrafficLight.YELLOW
        elif (n_green > threshould):
            return TrafficLight.GREEN
        else:
            return TrafficLight.UNKNOWN
        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image_light = self.detect_light(image)
        state_light = self.classify_light_state(image_light)
        return state_light


if __name__ == "__main__":
    model_name = 'ssd_mobilenet_v2_coco_2018_03_29'
    classifier = TLClassifier(model_name)

    cv_image = cv2.imread('imgs/0.jpg')
    # classifier.detect_light(cv_image)
    state = classifier.get_classification(cv_image)
