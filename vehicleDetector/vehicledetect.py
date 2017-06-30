#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 20:49:03 2017

@author: srikanthnarayanan
"""

import warnings
import cv2
import numpy as np
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from collections import deque
import sys
sys.path.append(r"../../CarND-Advanced-Lane-Lines/")
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Detector(object):
    '''
    A class that contains the definition of method and parameters needed for a
    vehicle detection in a given image
    '''
    def __init__(self, classifier, X_scaler, fp_threshold=1,
                 prob_detect_thresh=0.99, win_start=320, win_incr=16,
                 colourspace='YUV', smoothing=False, lanefinding=False,
                 lanedetector=None):
        '''
        Constructor for the Dectector object. Expect as MLP classifier and
        sklearn standard scaler fit object.
        '''
        self.clf = joblib.load(classifier)
        self.X_scaler = joblib.load(X_scaler)
        self.fp_threshold = fp_threshold
        self.prob_detect_thresh = prob_detect_thresh
        self.win_start = win_start
        self.colourspace = colourspace
        self.win_incr = win_incr
        self.vehiclebox = vehicleBox()
        self.smoothing = smoothing
        self.lanefinding = lanefinding
        self.lanedetector = lanedetector

    def extract_hog(self, img, color_space='RGB', orient=9, pix_per_cell=8,
                    cell_per_block=2, block_stride=8):
        '''
        Method to get Hog features of a given image. Excepts a RGB numpy array.
        '''
        if (img.shape[0], img.shape[1]) != (64, 64):
            img = cv2.resize(img, (64, 64))
        # Apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = img.copy()

        hog = cv2.HOGDescriptor((img.shape[1], img.shape[0]),
                                (pix_per_cell * cell_per_block,
                                 pix_per_cell * cell_per_block),
                                (block_stride, block_stride),
                                (pix_per_cell, pix_per_cell), orient)
        hog_features = np.ravel(hog.compute(feature_image))
        return hog_features

    def _draw_labeled_bboxes(self, labels, box_text=''):
        '''
        Helper method to draw rectangle around a given box and write a name on
        top.
        '''
        img = np.copy(self.image)
        box_list = []
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                    (np.max(nonzerox), np.max(nonzeroy)))
            box_list.append([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]])

            if not self.smoothing:
                # Draw the box on the image
                if self.lanefinding:
                    cv2.rectangle(self.lanedetector.temp_image, bbox[0],
                                  bbox[1], (0, 255, 0), 3)
                else:
                    cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 3)

        if self.smoothing:
            self.vehiclebox.oldboxlist.append(box_list)
            combined_box_list = np.ravel(self.vehiclebox.oldboxlist)
            unseperated_list = []
            i = 0
            while i <= len(combined_box_list)-3:
                unseperated_list.append(combined_box_list[i:i+4])
                i += 4

            # group previous rectangles toGether
            rects, wgts = cv2.groupRectangles(np.array(unseperated_list).
                                              tolist(), 1)
            for rect in rects:
                cv2.rectangle(img, (rect[0], rect[1]),
                                   (rect[2], rect[3]),
                                   (0, 255, 0), 5)
                cv2.putText(img, box_text, (rect[0], rect[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),
                            thickness=2)

        # Return the image
        if self.lanefinding:
            return self.lanedetector.temp_image
        else:
            return img

    def _apply_threshold(self, heatmap, threshold):
        '''
        A helper method to apply threshold
        '''
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def _add_heat(self, heatmap, bbox_list):
        '''
        Helper method to generate a heat map
        '''
        # Iterate through list of bboxes
        for box in bbox_list:
            for rect in box:
                # Add += 1 for all pixels inside each bbox
                # Assuming each "box" takes the form ((x1, y1), (x2, y2))
                heatmap[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]] += 1
        # Return updated heatmap
        return heatmap

    def slide_window(self, img, x_start_stop=[None, None],
                     y_start_stop=[None, None], xy_window=(64, 64),
                     xy_overlap=(0.5, 0.5)):
        '''
        A method to identify the sliding window x and y cordinates as tuple
        pair over the entire given image.
        '''
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] is None:
            x_start_stop[0] = 0
        if x_start_stop[1] is None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] is None:
            y_start_stop[0] = 0
        if y_start_stop[1] is None:
            y_start_stop[1] = img.shape[0]

        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]

        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

        # Compute the number of windows in x/y
        nx_windows = np.int(xspan / nx_pix_per_step)
        ny_windows = np.int(yspan / ny_pix_per_step)

        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = (xs + 1)*nx_pix_per_step + x_start_stop[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = (ys + 1) * ny_pix_per_step + y_start_stop[0]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def process_main(self, image):
        '''
        A main method to process a given image to identify the if vehicle is
        detected and add a rectangle around a detected image
        '''
        self.image = image
        detected = []
        dect_img = self.image.copy()
        win_size = self.win_start  # Intial Window start y value

        while win_size < image.shape[0]:
            windows = self.slide_window(dect_img, x_start_stop=[None, None],
                                        y_start_stop=[400, 660],
                                        xy_window=(win_size, win_size),
                                        xy_overlap=(0.8, 0.8))
            # predict for each window if it has a car
            for window in windows:
                curr_img = dect_img[window[0][1]: window[1][1],
                                    window[0][0]: window[1][0]]
                hog_feat = self.extract_hog(curr_img,
                                            color_space=self.colourspace)
                scaled_x = self.X_scaler.transform(hog_feat)
                if curr_img.shape[0] > 0:
                    pprob = self.clf.predict_proba(scaled_x.reshape(1, -1))
                    if pprob[0][1] > self.prob_detect_thresh:
                        detected.append(window)
            win_size = win_size + self.win_incr

        # If lane detection is on run lane detection process
        if self.lanefinding:
            if self.lanedetector is not None:
                lane_img = self.image.copy()
                self.lanedetector.runpipeline(lane_img)
        # generate heat map
        heat = np.zeros_like(self.image[:, :, 0]).astype(np.float)

        # Add heat to each box in box list
        if len(detected) > 0:
            self.vehiclebox.boxlist.append(detected)
        heat = self._add_heat(heat, self.vehiclebox.boxlist)

        # Apply threshold to help remove false positives
        heat = self._apply_threshold(heat, self.fp_threshold +
                                     len(self.vehiclebox.boxlist) // 2)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        # Final Box draw vehicle detected image
        draw_img = self._draw_labeled_bboxes(labels)

        return draw_img


class vehicleBox:
    '''
    A vehicle box class to hold the previous box shapes and smooth the box
    draw over frames
    '''
    def __init__(self):
        '''
        Constructor for vehicleBox double ended queue
        '''
        self.oldboxlist = deque(maxlen=2)
        self.boxlist = deque(maxlen=8)

if __name__ == "__main__":
    pass
