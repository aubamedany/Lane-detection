import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
from collections import deque
import math
# Needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
IMG_SIZE = [640, 360]
# %matplotlib inline
ym_per_pix = 30/360
        # meters per pixel in x dimension
xm_per_pix = 3.7/600

class Camera():    
    def __init__(self,nx=9,ny=6):
        # Stores the source 
        self.ret = None
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.nx = nx
        self.ny = ny
        self.objpoints = [] # 3D points in real space
        self.imgpoints = [] # 2D points in img space
        self.cal_images =[]
        
    def calibrate_camera(self, imgList):
        counter = 0
        for img in imgList:
            img = cv2.resize(img,IMG_SIZE)
            # Prepare object points (0,0,0), (1,0,0), etc.
            objp = np.zeros((self.nx*self.ny,3), np.float32)
            objp[:,:2] = np.mgrid[0:self.nx,0:self.ny].T.reshape(-1,2)

            # Converting to grayscale
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Finding chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
            if ret == True:
                self.imgpoints.append(corners)
                self.objpoints.append(objp)
                counter+=1
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
        return self.mtx, self.dist

    
    def undistort(self, img):
        return cv2.undistort(img,self.mtx,self.dist,None,self.mtx)
    def run_calibrate(self):
        cal_filenames = glob.glob("{}/*".format("camera_cal"))
        for f in cal_filenames:
            img = cv2.imread(f)
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            img = cv2.resize(img,IMG_SIZE)
            self.cal_images.append(np.array(img))
        mtx, dist = self.calibrate_camera(self.cal_images)

class Line():
    def __init__(self, maxSamples=4):
        
        self.maxSamples = maxSamples 
        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=self.maxSamples)
        # Polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        # Polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        # Average x values of the fitted line over the last n iterations
        self.bestx = None
        # Was the line detected in the last iteration?
        self.detected = False 
        # Radius of curvature of the line in some units
        self.radius_of_curvature = None 
        # Distance in meters of vehicle center from the line
        self.line_base_pos = None 
        
    def update_lane(self, ally, allx):
        # Updates lanes on every new frame
        # Mean x value 
        self.bestx = np.mean(allx, axis=0)
        # Fit 2nd order polynomial
        new_fit = np.polyfit(ally, allx, 2)
        # Update current fit3
        self.current_fit = new_fit
        # Add the new fit to the queue
        self.recent_xfitted.append(self.current_fit)
        # Use the queue mean as the best fit
        self.best_fit = np.mean(self.recent_xfitted, axis=0)
        # meters per pixel in y dimension
        
        # Calculate radius of curvature
        fit_cr = np.polyfit(ally*ym_per_pix, allx*xm_per_pix, 2)
        y_eval = np.max(ally)
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])


class Preprocess():
    def __init__(self):
        self.camera = Camera()
        self.src = np.float32([[20, 300], [50, 180], [400, 180], [600, 300]])
        self.offset = [75,0]
        self.dst = np.float32([self.src[0] + self.offset, np.array([self.src[0, 0], 0]) + self.offset, 
                            np.array([self.src[3, 0], 0]) - self.offset, self.src[3] - self.offset])
        self.left_line = Line()
        self.right_line = Line()
    def pers_transform(self,img, nx=9, ny=6):
        # Grab the image shape
        img_size = (img.shape[1], img.shape[0])
        # src = np.float32([[190, 720], [582, 457], [701, 457], [1145, 720]])
        # offset = [150,0]
        # dst = np.float32([src[0] + offset, np.array([src[0, 0], 0]) + offset, 
        #                   np.array([src[3, 0], 0]) - offset, src[3] - offset])

        # src = np.float32([[95, 360], [291, 228], [350, 228], [572, 360]])
        # offset = [75,0]
        # dst = np.float32([src[0] + offset, np.array([src[0, 0], 0]) + offset, 
        #                     np.array([src[3, 0], 0]) - offset, src[3] - offset])

        # project1
        # src = np.float32([[0, 360], [300, 150], [450, 150], [600, 360]])
        # dst = np.float32([[50,360],[50,0],[500,0],[500,360]])
        # project2
        # src = np.float32([[20, 300], [50, 180], [400, 180], [600, 300]])
        
        # dst = np.float32([[50,360],[50,0],[500,0],[500,360]])
        # offset = [75,0]
        # dst = np.float32([src[0] + offset, np.array([src[0, 0], 0]) + offset, 
        #                     np.array([src[3, 0], 0]) - offset, src[3] - offset])
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, img_size)
        # Return the resulting image and matrix
        Minv = cv2.getPerspectiveTransform(self.dst, self.src)

        return warped, M, Minv


    def hls_thresh(self,img, thresh_min=200, thresh_max=255):
        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,1]
        
        # Creating image masked in S channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh_min) & (s_channel <= thresh_max)] = 1
        return s_binary


    def sobel_thresh(self,img, sobel_kernel=3, orient='x', thresh_min=20, thresh_max=100):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
            abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
            scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        else:
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in x
            abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
            scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))
        
        # Creathing img masked in x gradient
        grad_bin = np.zeros_like(scaled_sobel)
        grad_bin[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        
        return grad_bin


    def mag_thresh(self,img, sobel_kernel=3, thresh_min=100, thresh_max=255):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= thresh_min) & (gradmag <= thresh_max)] = 1

        # Return the binary image
        return binary_output


    def dir_thresh(self,img, sobel_kernel=3, thresh_min=0, thresh_max=np.pi/2):
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh_min) & (absgraddir <= thresh_max)] = 1

        # Return the binary image
        return binary_output


    def lab_b_channel(self,img, thresh=(190,255)):
        # Normalises and thresholds to the B channel
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        lab_b = lab[:,:,2]
        # Don't normalize if there are no yellows in the image
        if np.max(lab_b) > 175:
            lab_b = lab_b*(255/np.max(lab_b))
        #  Apply a threshold
        binary_output = np.zeros_like(lab_b)
        binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
        return binary_output
    def mask_image(self,image):
        masked_image = np.copy(image)
        mask = np.zeros_like(masked_image)
        vertices = np.array([[self.src[1], self.src[0], self.src[2], self.src[3]]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, [255,255,255])
        masked_edges = cv2.bitwise_and(masked_image, mask)
        return masked_edges
    def test_process(self,img):
        
        # Undistorting image
        undist = self.camera.undistort(img)
        
        # Masking image
        masked = self.mask_image(undist)
        
        # Perspective transform image
        warped, M, Minv = self.pers_transform(undist)
        
        # Colour thresholding in S channel
        s_bin = self.hls_thresh(warped)
        
        # Colour thresholding in B channel of LAB
        b_bin = self.lab_b_channel(warped, thresh = (185, 255))
        
        # Gradient thresholding with sobel x
        x_bin = self.sobel_thresh(warped, orient='x', thresh_min=20, thresh_max=100)
        
        # Gradient thresholding with sobel y
        y_bin = self.sobel_thresh(warped, orient='y', thresh_min=50, thresh_max=150)
        
        # Magnitude of gradient thresholding
        mag_bin = self.mag_thresh(warped, thresh_min=0, thresh_max=255)
        
        # Direction of gradient thresholding
        dir_bin = self.dir_thresh(warped, thresh_min=0, thresh_max=np.pi/2)
        
        # Combining both thresholds
        combined = np.zeros_like(x_bin)
        combined[(s_bin==1) | (b_bin == 1)] = 1
        
        return combined, warped, Minv
    def window_search(self,binary_warped):
        # Take a histogram of the bottom half of the image
        bottom_half_y = binary_warped.shape[0]/2
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Generate black image and colour lane lines
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 1]
            
        # Draw polyline on image
        right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
        left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
        cv2.polylines(out_img, [right], False, (1,1,0), thickness=5)
        cv2.polylines(out_img, [left], False, (1,1,0), thickness=5)
        
        result = dict()
        result['left_lane_inds'] = left_lane_inds
        result['right_lane_inds'] = right_lane_inds
        result['out_img'] = out_img
        return result
        return left_lane_inds, right_lane_inds, out_img
    def margin_search(self,binary_warped):
        # Performs window search on subsequent frame, given previous frame.

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 30

        left_lane_inds = ((nonzerox > (self.left_line.current_fit[0]*(nonzeroy**2) + self.left_line.current_fit[1]*nonzeroy + self.left_line.current_fit[2] - margin)) & (nonzerox < (self.left_line.current_fit[0]*(nonzeroy**2) + self.left_line.current_fit[1]*nonzeroy + self.left_line.current_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (self.right_line.current_fit[0]*(nonzeroy**2) + self.right_line.current_fit[1]*nonzeroy + self.right_line.current_fit[2] - margin)) & (nonzerox < (self.right_line.current_fit[0]*(nonzeroy**2) + self.right_line.current_fit[1]*nonzeroy + self.right_line.current_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Generate a blank image to draw on
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        # Create an image to draw on and an image to show the selection window
        window_img = np.zeros_like(out_img)

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.intc([left_line_pts]), (0,255,0))
        cv2.fillPoly(window_img, np.intc([right_line_pts]), (0,255,0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] =  [0, 0, 255]
            
        # Draw polyline on image
        right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
        left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
        cv2.polylines(out_img, [right], False, (1,1,0), thickness=5)
        cv2.polylines(out_img, [left], False, (1,1,0), thickness=5)

        ret = {}
        ret['leftx'] = leftx
        ret['rightx'] = rightx
        ret['left_fitx'] = left_fitx
        ret['right_fitx'] = right_fitx
        ret['ploty'] = ploty
        ret['result'] = out_img
        ret['left_lane_inds'] = left_lane_inds
        ret['right_lane_inds'] = right_lane_inds 
        ret['out_img'] = out_img
        return ret
        return left_lane_inds, right_lane_inds, out_img
    def validate_lane_update(self,img, left_lane_inds, right_lane_inds):
        # Checks if detected lanes are good enough before updating
        img_size = (img.shape[1], img.shape[0])
        
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Extract left and right line pixel positions
        left_line_allx = nonzerox[left_lane_inds]
        left_line_ally = nonzeroy[left_lane_inds] 
        right_line_allx = nonzerox[right_lane_inds]
        right_line_ally = nonzeroy[right_lane_inds]
        
        # Discard lane detections that have very little points, 
        # as they tend to have unstable results in most cases
        if len(left_line_allx) <= 900 or len(right_line_allx) <= 900:
            self.left_line.detected = False
            self.right_line.detected = False
            return
        
        left_x_mean = np.mean(left_line_allx, axis=0)
        right_x_mean = np.mean(right_line_allx, axis=0)
        lane_width = np.subtract(right_x_mean, left_x_mean)
        
        # Discard the detections if lanes are not in their repective half of their screens
        if left_x_mean > 370 or right_x_mean < 370:
            self.left_line.detected = False
            self.right_line.detected = False
            return
        
        # Discard the detections if the lane width is too large or too small
        if  lane_width < 150 or lane_width > 400:
            self.left_line.detected = False
            self.right_line.detected = False
            return 
        
        # If this is the first detection or 
        # the detection is within the margin of the averaged n last lines 
        if self.left_line.bestx is None or np.abs(np.subtract(self.left_line.bestx, np.mean(left_line_allx, axis=0))) < 100:
            self.left_line.update_lane(left_line_ally, left_line_allx)
            self.left_line.detected = True
        else:
            self.left_line.detected = False
        if self.right_line.bestx is None or np.abs(np.subtract(self.right_line.bestx, np.mean(right_line_allx, axis=0))) < 100:
            self.right_line.update_lane(right_line_ally, right_line_allx)
            self.right_line.detected = True
        else:
            self.right_line.detected = False    
    
        # Calculate vehicle-lane offset
        # xm_per_pix = 3.7/610 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
        car_position = img_size[0]/2
        l_fit = self.left_line.current_fit
        r_fit = self.right_line.current_fit
        left_lane_base_pos = l_fit[0]*img_size[1]**2 + l_fit[1]*img_size[1] + l_fit[2]
        right_lane_base_pos = r_fit[0]*img_size[1]**2 + r_fit[1]*img_size[1] + r_fit[2]
        lane_center_position = (left_lane_base_pos + right_lane_base_pos) /2
        self.left_line.line_base_pos = (car_position - lane_center_position) * xm_per_pix +0.2
        self.right_line.line_base_pos = self.left_line.line_base_pos
    def find_lanes(self,img):
        if self.left_line.detected and self.right_line.detected:  # Perform margin search if exists prior success.
            # Margin Search
            ret = self.margin_search(img)
            left_lane_inds, right_lane_inds,out_img = ret['left_lane_inds'],ret["right_lane_inds"],ret["out_img"]
            # Update the lane detections
            self.validate_lane_update(img, left_lane_inds, right_lane_inds)
            
        else:  # Perform a full window search if no prior successful detections.
            # Window Search
            res =self.window_search(img)
            left_lane_inds, right_lane_inds,out_img = res["left_lane_inds"],res["right_lane_inds"],res["out_img"]
            # Update the lane detections
            self.validate_lane_update(img, left_lane_inds, right_lane_inds)
        return out_img


    def write_stats(self,img):
        try:
            font = cv2.FONT_HERSHEY_PLAIN
            size = 1.5
            weight = 2
            color = (255,255,255)
            
            radius_of_curvature = (self.left_line.radius_of_curvature + self.right_line.radius_of_curvature)/2.0
            cv2.putText(img,'Lane Curvature Radius: '+ '{0:.2f}'.format(radius_of_curvature)+'m',(15,15), font, size, color, weight)

            # if (self.left_line.line_base_pos >=0):
            #     cv2.putText(img,'Vehicle is '+ '{0:.2f}'.format(self.left_line.line_base_pos*100)+'cm'+ ' Right of Center',(15,50), font, size, color, weight)
            # else:
            #     cv2.putText(img,'Vehicle is '+ '{0:.2f}'.format(abs(self.left_line.line_base_pos)*100)+'cm' + ' Left of Center',(15,50), font, size, color, weight)
        except Exception as e:
            print(e)
            
    def draw_lane(self,undist, img, Minv):
        # Generate x and y values for plotting
        ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(img).astype(np.uint8)
        color_warp = np.stack((warp_zero, warp_zero, warp_zero), axis=-1)

        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit
        if left_fit is not None and right_fit is not None:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            
            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            
            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
            
            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
            
            # Combine the result with the original image
            result = cv2.addWeighted(undist, 1, newwarp, 0.6, 0)
            self.write_stats(result)
            return result
        return undist
    def assemble_img(self,warped, threshold_img, polynomial_img, lane_img):
        # Define output image
        font = cv2.FONT_HERSHEY_PLAIN
        size = 1.5
        weight = 2
        color = (255,255,255)
        res = self.get_steer_angle(lane_img)
        direction = res["direction"]
        angle = res["angle"]
        cv2.putText(lane_img,'Steer Angel: '+ '{0:.6f}'.format(angle)+' degree',(15,30), font, size, color, weight)
        cv2.putText(lane_img,'Direction: '+ direction,(15,50), font, size, color, weight)
        return lane_img
        # Main image
        img_out=np.zeros((360,854,3), dtype=np.uint8)
        # img_out[0:720,0:1280,:] = lane_img
        img_out[0:360,0:640,:] = lane_img
        
        # Text formatting
        fontScale=1
        thickness=1
        fontFace = cv2.FONT_HERSHEY_PLAIN
        
        # Perspective transform image
        img_out[0:120,641:854,:] = cv2.resize(warped,(213,120))
        boxsize, _ = cv2.getTextSize("Transformed", fontFace, fontScale, thickness)
        cv2.putText(img_out, "Transformed", (int(747-boxsize[0]/2),40), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)
    
        # Threshold image
        resized = cv2.resize(threshold_img,(213,120))
        resized=np.uint8(resized)
        gray_image = cv2.cvtColor(resized*255,cv2.COLOR_GRAY2RGB)
        img_out[121:241,641:854,:] = cv2.resize(gray_image,(213,120))
        boxsize, _ = cv2.getTextSize("Filtered", fontFace, fontScale, thickness)
        cv2.putText(img_out, "Filtered", (int(747-boxsize[0]/2),141), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)
    
        # Polynomial lines
        img_out[240:360,641:854,:] = cv2.resize(polynomial_img*255,(213,120))
        boxsize, _ = cv2.getTextSize("Detected Lanes", fontFace, fontScale, thickness)
        cv2.putText(img_out, "Detected Lanes", (int(747-boxsize[0]/2),523621), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)
        


        return img_out

    def process_img(self,img):

        img =cv2.resize(img,IMG_SIZE)
        # Undistorting image
        undist = self.camera.undistort(img)
        # Masking image
        masked = self.mask_image(undist)
        
        # Perspective transform image
        warped, M, Minv = self.pers_transform(undist)
        
        # Colour thresholding in S channel
        s_bin = self.hls_thresh(warped)
        
        # Colour thresholding in B channel of LAB
        b_bin = self.lab_b_channel(warped, thresh = (185, 255))
        
        # Combining both thresholds
        combined = np.zeros_like(s_bin)
        combined[(s_bin==1) | (b_bin == 1)] = 1
        
        # Find Lanes
        output_img = self.find_lanes(combined)

        # Draw lanes on image
        lane_img = self.draw_lane(undist, combined, Minv); 
        
        result = self.assemble_img(warped, combined, output_img, lane_img)    
        
        return result

    def angleCalculator(self, x, y):
        slope = (x - 72) / float(y - 144) # (320, 360) is center of (640, 360) image
        angleRadian = float(math.atan(slope))
        angleDegree = float(angleRadian * 180.0 / math.pi)
        return angleDegree
    
    def computeCenter(self, roadImg):
        count = 0
        center_x = 0
        center_y = 0 
        gray_img = cv2.cvtColor(roadImg, cv2.COLOR_BGR2GRAY)

        for i in range(0,144):
            for j in range(0,144):
                if gray_img[i][j] == 255:
                    count += 1
                    center_x += i
                    center_y += j
        

        if center_x != 0 or center_y != 0 or count != 0:
            center_x = center_x / count
            center_x = center_y / count
            angleDegree = self.angleCalculator(center_x, center_y) # Call angle_calculator method in speed_up.py to use numba function

        return angleDegree
    def get_steer_angle(self,img):
        angle = self.computeCenter(img)
        direction = "left" if (self.left_line.line_base_pos >=0) else "right"
        res = dict()
        res["angle"] = angle
        res["direction"] = direction
        return res