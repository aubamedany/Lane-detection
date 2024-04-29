import cv2 # Import the OpenCV library to enable computer vision
import numpy as np # Import the NumPy scientific computing library
import edge_detection as edge # Handles the detection of lane lines
import matplotlib.pyplot as plt # Used for plotting and error checking
import glob
import math
IMG_SIZE = (640,360)
THRESHOLD_ANGLE = 1.5
filename = 'camera_front_view2.mp4'
def calculate_angle(base=90, offset=1):
   angleRadian=float(math.atan(offset/base))
   angleDegree = float(angleRadian * 180.0 / math.pi)
   return angleDegree
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
        self.angle = None
        self.direction =None
        
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


class Lane:
  """
  Represents a lane on a road.
  """
  def __init__(self, orig_frame):
    """
      Default constructor
         
    :param orig_frame: Original camera image (i.e. frame)
    """
    self.orig_frame = orig_frame
    self.camera = Camera()
    # This will hold an image with the lane lines       
    self.lane_line_markings = None
 
    # This will hold the image after perspective transformation
    self.warped_frame = None
    self.transformation_matrix = None
    self.inv_transformation_matrix = None
 
    # (Width, Height) of the original video frame (or image)
    self.orig_image_size = self.orig_frame.shape[::-1][1:]
 
    width = self.orig_image_size[0]
    height = self.orig_image_size[1]
    self.width = width
    self.height = height
     
    # Four corners of the trapezoid-shaped region of interest
    # You need to find these corners manually.

    # self.roi_points = np.float32([
    #   (274,184), # Top-left corner
    #   (0, 337), # Bottom-left corner            
    #   (575,337), # Bottom-right corner
    #   (371,184) # Top-right corner
    # ])
         
    # self.roi_points = np.float32([
    #   (170,200), # Top-left corner
    #   (0, 360), # Bottom-left corner            
    #   (570,360), # Bottom-right corner
    #   (430,200) # Top-right corner
    # ])
    self.roi_points = np.float32([
      (200,180), # Top-left corner
      (75, 290), # Bottom-left corner            
      (510,290), # Bottom-right corner
      (415,180) # Top-right corner
    ])  

    self.padding = int(0.25 * width) # padding from side of the image in pixels
    # self.padding = 100
    self.desired_roi_points = np.float32([
      [self.padding, 0], # Top-left corner
      [self.padding, self.orig_image_size[1]], # Bottom-left corner         
      [self.orig_image_size[
        0]-self.padding, self.orig_image_size[1]], # Bottom-right corner
      [self.orig_image_size[0]-self.padding, 0] # Top-right corner
    ]) 
         
    # Histogram that shows the white pixel peaks for lane line detection
    self.histogram = None
         
    # Sliding window parameters
    self.no_of_windows = 10
    self.margin = int((1/12) * width)  # Window width is +/- margin
    self.minpix = int((1/24) * width)  # Min no. of pixels to recenter window
         
    # Best fit polynomial lines for left line and right line of the lane
    self.left_fit = None
    self.right_fit = None
    self.left_lane_inds = None
    self.right_lane_inds = None
    self.ploty = None
    self.left_fitx = None
    self.right_fitx = None
    self.leftx = None
    self.rightx = None
    self.lefty = None
    self.righty = None
    self.birdeye = None
    self.sliding_window = None
    # Pixel parameters for x and y dimensions
    self.YM_PER_PIX = 10.0 / 1000 # meters per pixel in y dimension
    self.XM_PER_PIX = 3.7 / 781 # meters per pixel in x dimension
         
    # Radii of curvature and offset
    self.left_curvem = None
    self.right_curvem = None
    self.center_offset = None
    self.lane_center =None
    self.car_pos =None
    self.res = None
    self.overlay =None
  def calculate_car_position(self, print_to_terminal=False):

    car_location = self.orig_frame.shape[1] / 2
 
    # Fine the x coordinate of the lane line bottom
    # height = self.orig_frame.shape[0]/2
  
    # bottom_left = self.left_fit[0]*height**2 + self.left_fit[
    #   1]*height + self.left_fit[2]
    # bottom_right = self.right_fit[0]*height**2 + self.right_fit[
    #   1]*height + self.right_fit[2]
 
    # center_lane = (bottom_right + bottom_left)/2 
    

    y = np.linspace(
      75, 149, 150)
    left= self.left_fit[0]*y**2 + self.left_fit[1]*y + self.left_fit[2]
    right = self.right_fit[0]*y**2 + self.right_fit[1]*y + self.right_fit[2]

    center_lane = ((np.sum(left)/150)+(np.sum(right)/150))//2
    # center_lane = ((np.sum(self.left_fitx)/360)+(np.sum(self.right_fitx)/360))//2

    self.lane_center = center_lane -18
    self.car_pos = car_location
    center_offset = (np.abs(self.car_pos) - np.abs(
      self.lane_center))
    self.angle = calculate_angle(offset =center_offset )
    self.angle = 0 if abs(self.angle)<THRESHOLD_ANGLE else self.angle
    self.direction = 'Left' if self.angle >=0 else "Right"
    if print_to_terminal == True:
      print(str(center_offset) + 'cm')
             
    self.center_offset = center_offset
       
    return center_offset
 
  def calculate_curvature(self, print_to_terminal=False):
    """
    Calculate the road curvature in meters.
 
    :param: print_to_terminal Display data to console if True
    :return: Radii of curvature
    """
    # Set the y-value where we want to calculate the road curvature.
    # Select the maximum y-value, which is the bottom of the frame.
    y_eval = np.max(self.ploty)    
 
    # Fit polynomial curves to the real world environment
    left_fit_cr = np.polyfit(self.lefty * self.YM_PER_PIX, self.leftx * (
      self.XM_PER_PIX), 2)
    right_fit_cr = np.polyfit(self.righty * self.YM_PER_PIX, self.rightx * (
      self.XM_PER_PIX), 2)
             
    # Calculate the radii of curvature
    left_curvem = ((1 + (2*left_fit_cr[0]*y_eval*self.YM_PER_PIX + left_fit_cr[
                    1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curvem = ((1 + (2*right_fit_cr[
                    0]*y_eval*self.YM_PER_PIX + right_fit_cr[
                    1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
     
    # Display on terminal window
    if print_to_terminal == True:
      print(left_curvem, 'm', right_curvem, 'm')
             
    self.left_curvem = left_curvem
    self.right_curvem = right_curvem
 
    return left_curvem, right_curvem        
         
  def calculate_histogram(self,frame=None,plot=True):
    """
    Calculate the image histogram to find peaks in white pixel count
         
    :param frame: The warped image
    :param plot: Create a plot if True
    """
    if frame is None:
      frame = self.warped_frame
             
    # Generate the histogram
    self.histogram = np.sum(frame[int(
              frame.shape[0]/2):,:], axis=0)
 
    if plot == True:
         
      # Draw both the image and the histogram
      figure, (ax1, ax2) = plt.subplots(2,1) # 2 row, 1 columns
      figure.set_size_inches(10, 5)
      ax1.imshow(frame, cmap='gray')
      ax1.set_title("Warped Binary Frame")
      ax2.plot(self.histogram)
      ax2.set_title("Histogram Peaks")
      plt.show()
             
    return self.histogram
 
  def display_curvature_offset(self, frame=None, plot=False):
    """
    Display curvature and offset statistics on the image
         
    :param: plot Display the plot if True
    :return: Image with lane lines and curvature
    """

    image_copy = None
    if frame is None:
      image_copy = self.orig_frame.copy()
    else:
      image_copy = frame
 
    cv2.putText(image_copy,'Curve Radius: '+str((
      self.left_curvem+self.right_curvem)/2)[:7]+' m', (int((
      5/600)*self.width), int((
      20/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, (float((
      0.5/600)*self.width)),(
      255,255,255),2,cv2.LINE_AA)
    cv2.putText(image_copy,'Center Offset: '+str(
      self.center_offset)[:7]+' cm', (int((
      5/600)*self.width), int((
      40/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, (float((
      0.5/600)*self.width)),(
      255,255,255),2,cv2.LINE_AA)
             
    if plot==True:       
      cv2.imshow("Image with Curvature and Offset", image_copy)
 
    return image_copy
     
  def get_lane_line_previous_window(self, left_fit, right_fit, plot=False):

    margin = self.margin
 
    # Find the x and y coordinates of all the nonzero 
    # (i.e. white) pixels in the frame.         
    nonzero = self.warped_frame.nonzero()  
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
         
    # Store left and right lane pixel indices
    left_lane_inds = ((nonzerox > (left_fit[0]*(
      nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (
      nonzerox < (left_fit[0]*(
      nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(
      nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (
      nonzerox < (right_fit[0]*(
      nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))           
    self.left_lane_inds = left_lane_inds
    self.right_lane_inds = right_lane_inds
 
    # Get the left and right lane line pixel locations  
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]  
 
    self.leftx = leftx
    self.rightx = rightx
    self.lefty = lefty
    self.righty = righty        
     
    # Fit a second order polynomial curve to each lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    self.left_fit = left_fit
    self.right_fit = right_fit
         
    # Create the x and y values to plot on the image
    ploty = np.linspace(
      0, self.warped_frame.shape[0]-1, self.warped_frame.shape[0]) 
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    self.ploty = ploty
    self.left_fitx = left_fitx
    self.right_fitx = right_fitx
    if plot==True:
         
      # Generate images to draw on
        out_img = np.dstack((self.warped_frame, self.warped_frame, (
                            self.warped_frame)))*255
        window_img = np.zeros_like(out_img)
                
        # Add color to the left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
                                                                        0, 0, 255]
        # Create a polygon to show the search window area, and recast 
        # the x and y points into a usable format for cv2.fillPoly()
        # margin = self.margin
        margin = 5
        left_line_window1 = np.array([np.transpose(np.vstack([
                                        left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                                        left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([
                                        right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                                        right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
                
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        # result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        self.birdeye = window_img
        # Plot the figures 
        # figure, (ax1, ax2, ax3) = plt.subplots(3,1) # 3 rows, 1 column
        # figure.set_size_inches(10, 10)
        # figure.tight_layout(pad=3.0)
        # ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
        # ax2.imshow(self.warped_frame, cmap='gray')
        # ax3.imshow(result)
        # ax3.plot(left_fitx, ploty, color='yellow')
        # ax3.plot(right_fitx, ploty, color='yellow')
        # ax1.set_title("Original Frame")  
        # ax2.set_title("Warped Frame")
        # ax3.set_title("Warped Frame With Search Window")
        # self.birdeye = result
        # plt.show()
             
  def get_lane_line_indices_sliding_windows(self, plot=False):
    """
    Get the indices of the lane line pixels using the 
    sliding windows technique.
         
    :param: plot Show plot or not
    :return: Best fit lines for the left and right lines of the current lane 
    """
    # Sliding window width is +/- margin
    margin = self.margin
 
    frame_sliding_window = self.warped_frame.copy()
 
    # Set the height of the sliding windows
    window_height = int(self.warped_frame.shape[0]/self.no_of_windows)       
 
    # Find the x and y coordinates of all the nonzero 
    # (i.e. white) pixels in the frame. 
    nonzero = self.warped_frame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1]) 
         
    # Store the pixel indices for the left and right lane lines
    left_lane_inds = []
    right_lane_inds = []
         
    # Current positions for pixel indices for each window,
    # which we will continue to update
    leftx_base, rightx_base = self.histogram_peak()
    leftx_current = leftx_base
    rightx_current = rightx_base
 
    # Go through one window at a time
    no_of_windows = self.no_of_windows
         
    for window in range(no_of_windows):
       
      # Identify window boundaries in x and y (and right and left)
      win_y_low = self.warped_frame.shape[0] - (window + 1) * window_height
      win_y_high = self.warped_frame.shape[0] - window * window_height
      win_xleft_low = leftx_current - margin
      win_xleft_high = leftx_current + margin
      win_xright_low = rightx_current - margin
      win_xright_high = rightx_current + margin
      cv2.rectangle(frame_sliding_window,(win_xleft_low,win_y_low),(
        win_xleft_high,win_y_high), (255,255,255), 2)
      cv2.rectangle(frame_sliding_window,(win_xright_low,win_y_low),(
        win_xright_high,win_y_high), (255,255,255), 2)
 
      # Identify the nonzero pixels in x and y within the window
      good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (
                           nonzerox < win_xleft_high)).nonzero()[0]
      good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (
                            nonzerox < win_xright_high)).nonzero()[0]
                                                         
      # Append these indices to the lists
      left_lane_inds.append(good_left_inds)
      right_lane_inds.append(good_right_inds)
         
      # If you found > minpix pixels, recenter next window on mean position
      minpix = self.minpix
      if len(good_left_inds) > minpix:
        leftx_current = int(np.mean(nonzerox[good_left_inds]))
      if len(good_right_inds) > minpix:        
        rightx_current = int(np.mean(nonzerox[good_right_inds]))
                     
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
 
    # Extract the pixel coordinates for the left and right lane lines
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds] 
    righty = nonzeroy[right_lane_inds]
 
    # Fit a second order polynomial curve to the pixel coordinates for
    # the left and right lane lines
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2) 
         
    self.left_fit = left_fit
    self.right_fit = right_fit
 
    if plot==True:
         
      # Create the x and y values to plot on the image  
      self.ploty = np.linspace(
        0, frame_sliding_window.shape[0]-1, frame_sliding_window.shape[0])
      self.left_fitx = left_fit[0]*self.ploty**2 + left_fit[1]*self.ploty + left_fit[2]
      self.right_fitx = right_fit[0]*self.ploty**2 + right_fit[1]*self.ploty + right_fit[2]
 
      # Generate an image to visualize the result
      out_img = np.dstack((
        frame_sliding_window, frame_sliding_window, (
        frame_sliding_window))) * 255
             
      # Add color to the left line pixels and right line pixels
      out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
      out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
        0, 0, 255]
                 
      # Plot the figure with the sliding windows
      # figure, (ax1, ax2, ax3) = plt.subplots(3,1) # 3 rows, 1 column
      # figure.set_size_inches(10, 10)
      # figure.tight_layout(pad=3.0)
      # ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
      # ax2.imshow(frame_sliding_window, cmap='gray')
      # ax3.imshow(out_img)
      # ax3.plot(left_fitx, ploty, color='yellow')
      # ax3.plot(right_fitx, ploty, color='yellow')
      # ax1.set_title("Original Frame")  
      # ax2.set_title("Warped Frame with Sliding Windows")
      # ax3.set_title("Detected Lane Lines with Sliding Windows")
      self.sliding_window = frame_sliding_window
      # plt.show()        
             
    return self.left_fit, self.right_fit
 
  def get_line_markings(self, frame=None):
   
    if frame is None:
      frame = self.orig_frame

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
 

    _, sxbinary = edge.threshold(hls[:, :, 1], thresh=(120, 255))
    sxbinary = edge.blur_gaussian(sxbinary, ksize=3) # Reduce noise

    sxbinary = edge.mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))
 

    s_channel = hls[:, :, 2] 
    _, s_binary = edge.threshold(s_channel, (0, 255))
     

    _, r_thresh = edge.threshold(frame[:, :, 2], thresh=(120, 255))
     
    rs_binary = cv2.bitwise_and(s_binary, r_thresh)

    self.lane_line_markings = cv2.bitwise_or(rs_binary, sxbinary.astype(
                              np.uint8))    
    return self.lane_line_markings
         
  def histogram_peak(self):
    """
    Get the left and right peak of the histogram
 
    Return the x coordinate of the left histogram peak and the right histogram
    peak.
    """
    midpoint = int(self.histogram.shape[0]/2)
    leftx_base = np.argmax(self.histogram[:midpoint])
    rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint
    

    # (x coordinate of left peak, x coordinate of right peak)
    return leftx_base, rightx_base
         
  def overlay_lane_lines(self, plot=False):
    """
    Overlay lane lines on the original frame
    :param: Plot the lane lines if True
    :return: Lane with overlay
    """
    # Generate an image to draw the lane lines on 
    warp_zero = np.zeros_like(self.warped_frame).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))       

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([
                         self.left_fitx, self.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([
                          self.right_fitx, self.ploty])))])
    pts = np.hstack((pts_left, pts_right))
         
    # Draw lane on the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
 
    # Warp the blank back to original image space using inverse perspective 
    # matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, self.inv_transformation_matrix, (
                                  self.orig_frame.shape[
                                  1], self.orig_frame.shape[0]))
     
    # Combine the result with the original image
    result = cv2.addWeighted(self.orig_frame, 1, newwarp, 0.3, 0)
    self.overlay=color_warp
    self.res = result
    if plot==True:
      
      # Plot the figures 
      figure, (ax1, ax2) = plt.subplots(2,1) # 2 rows, 1 column
      figure.set_size_inches(10, 10)
      figure.tight_layout(pad=3.0)
      ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
      ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
      ax1.set_title("Original Frame")  
      ax2.set_title("Original Frame With Lane Overlay")
      plt.show()   
 
    return result           
     
  def perspective_transform(self, frame=None, plot=False):
    """
    Perform the perspective transform.
    :param: frame Current frame
    :param: plot Plot the warped image if True
    :return: Bird's eye view of the current lane
    """
    if frame is None:
      frame = self.lane_line_markings
             
    # Calculate the transformation matrix
    self.transformation_matrix = cv2.getPerspectiveTransform(
      self.roi_points, self.desired_roi_points)
 
    # Calculate the inverse transformation matrix           
    self.inv_transformation_matrix = cv2.getPerspectiveTransform(
      self.desired_roi_points, self.roi_points)
 
    # Perform the transform using the transformation matrix
    self.warped_frame = cv2.warpPerspective(
      frame, self.transformation_matrix, self.orig_image_size, flags=(
     cv2.INTER_LINEAR)) 
 
    # Convert image to binary
    (thresh, binary_warped) = cv2.threshold(
      self.warped_frame, 127, 255, cv2.THRESH_BINARY)           
    self.warped_frame = binary_warped

    # Display the perspective transformed (i.e. warped) frame
    if plot == True:
      warped_copy = self.warped_frame.copy()
      warped_plot = cv2.polylines(warped_copy, np.int32([
                    self.desired_roi_points]), True, (147,20,255), 3)
 
      # Display the image
      while(1):
        cv2.imshow('Warped Image', warped_plot)
             
        # Press any key to stop
        if cv2.waitKey(0):
          break
 
      cv2.destroyAllWindows()   
             
    return self.warped_frame        
     
  def plot_roi(self, frame=None, plot=False):
    """
    Plot the region of interest on an image.
    :param: frame The current image frame
    :param: plot Plot the roi image if True
    """
    if plot == False:
      return
             
    if frame is None:
      frame = self.orig_frame.copy()
 
    # Overlay trapezoid on the frame
    this_image = cv2.polylines(frame, np.int32([
      self.roi_points]), True, (147,20,255), 3)
 
    # Display the image
    while(1):
      cv2.imshow('ROI Image', this_image)
             
      # Press any key to stop
      if cv2.waitKey(0):
        break
 
    cv2.destroyAllWindows()
def pipeline(img):
    lane_obj = Lane(orig_frame=img)
    # img = lane_obj.camera.undistort(img)
    # Perform thresholding to isolate lane lines
    lane_line_markings = lane_obj.get_line_markings()
    
    # Plot the region of interest on the image
    lane_obj.plot_roi(plot=False)

    warped_frame = lane_obj.perspective_transform(plot=False)

    histogram = lane_obj.calculate_histogram(plot=False)  
    left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(
        plot=True)
    lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=True)
    frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=False)
    lane_obj.calculate_curvature(print_to_terminal=False)                                  
    lane_obj.calculate_car_position(print_to_terminal=False)
    frame_with_lane_lines2 = lane_obj.display_curvature_offset(
        frame=frame_with_lane_lines, plot=False)
    
    # size = len(filename)
    # new_filename = filename[:size - 4]
    # new_filename = new_filename + '_thresholded.jpg'    
    # result = lane_obj.res
    # font = cv2.FONT_HERSHEY_PLAIN
    # size = 1.5
    # weight = 2
    # color = (255,255,255)
    # color2 =(0,255,0)
    # cv2.putText(result,'.',(int(lane_obj.lane_center),300), font, size, color, weight) 
    # cv2.putText(result,'.',(int(lane_obj.car_pos),300), font, size, color2, weight) 
    font = cv2.FONT_HERSHEY_PLAIN
    size = 1.5
    weight = 2
    color = (255,255,255)
    color2 =(0,255,0)
    res = lane_obj.birdeye
    ori = lane_obj.res
    cv2.putText(res,'.',(int(lane_obj.lane_center),180), font, size, color2, weight) 
    cv2.putText(res,'.',(int(lane_obj.car_pos),180), font, size, color, weight) 
    cv2.putText(ori,f'Angle:{lane_obj.angle}',(10,70), font, size, color, weight)
    cv2.putText(ori,f'Direction:{lane_obj.direction}',(400,70), font, size, color, weight)
    cv2.polylines(ori, np.int32([
      lane_obj.roi_points]), True, (147,20,255), 3)
    cv2.imshow("test",ori)
    cv2.imshow("Bird eye",res) 
    cv2.imshow("warped",lane_obj.sliding_window)
    print(lane_obj.angle)
    # plt.imshow(img, cmap='gray')
def main():
    video = cv2.VideoCapture(filename) 
    video.set(cv2.CAP_PROP_FPS, 10)   
    camera =Camera()
    camera.run_calibrate()
    while (video.isOpened()):
        try:
            flag, img = video.read()
            img = cv2.resize(img,IMG_SIZE)
            img = camera.undistort(img)
            
            # img = cv2.resize(img, IMG_SIZE)
            # if not flag: break
            # b, g, r = cv2.split(img)
            # img = cv2.merge([r, g, b])
            pipeline(img)
            if cv2.waitKey(1) == ord('q'):break
        except Exception as e:
            print(e) 
# def main():
        
    # filename = "/Users/namle/BachKhoa/HK232/DOAN/Advanced-Lane-Lines/images/test_images/frame_2.jpg"
    # # Load a frame (or image)
    # original_frame = cv2.imread(filename)
    # # original_frame = cv2.resize(original_frame,(640,320))
    # # Create a Lane object
    # lane_obj = Lane(orig_frame=original_frame)
    
    # # Perform thresholding to isolate lane lines
    # lane_line_markings = lane_obj.get_line_markings()
    
    # # Plot the region of interest on the image
    # lane_obj.plot_roi(plot=False)
    
    # # Perform the perspective transform to generate a bird's eye view
    # # If Plot == True, show image with new region of interest
    # warped_frame = lane_obj.perspective_transform(plot=False)
    
    # # Generate the image histogram to serve as a starting point
    # # for finding lane line pixels
    # histogram = lane_obj.calculate_histogram(plot=False)  
        
    # # Find lane line pixels using the sliding window method 
    # left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(
    #     plot=True)
    # # Fill in the lane line
    # lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=False)
        
    # # Overlay lines on the original frame
    # frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=True)
    
    # # Calculate lane line curvature (left and right lane lines)
    # lane_obj.calculate_curvature(print_to_terminal=False)
    
    # # Calculate center offset                                                                 
    # lane_obj.calculate_car_position(print_to_terminal=False)
        
    # # Display curvature and center offset on image
    # frame_with_lane_lines2 = lane_obj.display_curvature_offset(
    #     frame=frame_with_lane_lines, plot=True)
        
    # # Create the output file name by removing the '.jpg' part
    # size = len(filename)
    # new_filename = filename[:size - 4]
    # new_filename = new_filename + '_thresholded.jpg'     
        
    # # Save the new image in the working directory
    # #cv2.imwrite(new_filename, lane_line_markings)
    
    # # Display the image 
    # # cv2.imshow("Image", lane_line_markings) 
        
    # # Display the window until any key is pressed
    # cv2.waitKey(0) 
        
    # # Close all windows
    # cv2.destroyAllWindows() 

main()