import glob
import os

import numpy as np
from PIL import Image
from scipy import signal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

# def display_array(a, rng=[0,1]):
#     a = (a - rng[0])/float(rng[1] - rng[0])*255
#     a = np.uint8(np.clip(a, 0, 255))
#     img = Image.fromarray(a, "L")
#     display(img)

def make_mask(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape([*a.shape, 1, 1])
  return tf.constant(a, dtype=1)

x_mask = make_mask([[-1.0, 0.0, 1.0],
                    [-1.0, 0.0, 1.0],
                    [-1.0, 0.0, 1.0]])

y_mask = make_mask([[-1.0,-1.0,-1.0],
                    [ 0.0, 0.0, 0.0],
                    [ 1.0, 1.0, 1.0]])

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def gradientx(x):
  """Compute the x gradient of an array"""
  return simple_conv(x, x_mask)

def gradienty(x):
  """Compute the x gradient of an array"""
  return simple_conv(x, y_mask)

def checkMatch(lineset):
    """Checks whether there exists 7 lines of consistent increasing order in set of lines"""
    linediff = np.diff(lineset)
    x = 0
    cnt = 0
    for line in linediff:
        # Within 5 px of the other (allowing for minor image errors)
        if np.abs(line - x) < 5:
            cnt += 1
        else:
            cnt = 0
            x = line
    return cnt == 5

def pruneLines(lineset):
    """Prunes a set of lines to 7 in consistent increasing order (chessboard)"""
    linediff = np.diff(lineset)
    x = 0
    cnt = 0
    start_pos = 0
    for i, line in enumerate(linediff):
        # Within 5 px of the other (allowing for minor image errors)
        if np.abs(line - x) < 5:
            cnt += 1
            if cnt == 5:
                end_pos = i+2
                return lineset[start_pos:end_pos]
        else:
            cnt = 0
            x = line
            start_pos = i
    return lineset

def skeletonize_1d(arr):
    """return skeletonized 1d array (thin to single value, favor to the right)"""
    _arr = arr.copy() # create a copy of array to modify without destroying original
    # Go forwards
    for i in range(_arr.size-1):
        # Will right-shift if they are the same
        if arr[i] <= _arr[i+1]:
            _arr[i] = 0

    # Go reverse
    for i in np.arange(_arr.size-1, 0,-1):
        if _arr[i-1] > _arr[i]:
            _arr[i] = 0
    return _arr

def getChessLines(hdx, hdy, hdx_thresh, hdy_thresh):
    """Returns pixel indices for the 7 internal chess lines in x and y axes"""
    # Blur
    gausswin = signal.gaussian(21,4)
    gausswin /= np.sum(gausswin)

    # Blur where there is a strong horizontal or vertical line (binarize)
    blur_x = np.convolve(hdx > hdx_thresh, gausswin, mode="same")
    blur_y = np.convolve(hdy > hdy_thresh, gausswin, mode="same")


    skel_x = skeletonize_1d(blur_x)
    skel_y = skeletonize_1d(blur_y)

    # Find points on skeletonized arrays (where returns 1-length tuple)
    lines_x = np.where(skel_x)[0] # vertical lines
    lines_y = np.where(skel_y)[0] # horizontal lines

    # Prune inconsistent lines
    lines_x = pruneLines(lines_x)
    lines_y = pruneLines(lines_y)

    is_match = len(lines_x) == 7 and len(lines_y) == 7 and checkMatch(lines_x) and checkMatch(lines_y)

    return lines_x, lines_y, is_match

def getChessTiles(a, lines_x, lines_y):
    """Split up input grayscale array into 64 tiles stacked in a 3D matrix using the chess linesets"""
    # Find average square size, round to a whole pixel for determining edge pieces sizes
    stepx = np.int32(np.round(np.mean(np.diff(lines_x))))
    stepy = np.int32(np.round(np.mean(np.diff(lines_y))))

    # Pad edges as needed to fill out chessboard (for images that are partially over-cropped)
    padr_x = 0
    padl_x = 0
    padr_y = 0
    padl_y = 0

    if lines_x[0] - stepx < 0:
        padl_x = np.abs(lines_x[0] - stepx)
    if lines_x[-1] + stepx > a.shape[1]-1:
        padr_x = np.abs(lines_x[-1] + stepx - a.shape[1])
    if lines_y[0] - stepy < 0:
        padl_y = np.abs(lines_y[0] - stepy)
    if lines_y[-1] + stepx > a.shape[0]-1:
        padr_y = np.abs(lines_y[-1] + stepy - a.shape[0])

    # New padded array
    a2 = np.pad(a, ((padl_y,padr_y),(padl_x,padr_x)), mode="edge")

    setsx = np.hstack([lines_x[0]-stepx, lines_x, lines_x[-1]+stepx]) + padl_x
    setsy = np.hstack([lines_y[0]-stepy, lines_y, lines_y[-1]+stepy]) + padl_y

    a2 = a2[setsy[0]:setsy[-1], setsx[0]:setsx[-1]]
    setsx -= setsx[0]
    setsy -= setsy[0]

    # Matrix to hold images of individual squares (in grayscale)
    squares = np.zeros([np.round(stepy), np.round(stepx), 64],dtype=np.uint8)

    # For each row
    for i in range(0,8):
        # For each column
        for j in range(0,8):
            # Vertical lines
            x1 = setsx[i]
            x2 = setsx[i+1]
            padr_x = 0
            padl_x = 0
            padr_y = 0
            padl_y = 0

            if (x2-x1) > stepx:
                if i == 7:
                    x1 = x2 - stepx
                else:
                    x2 = x1 + stepx
            elif (x2-x1) < stepx:
                if i == 7:
                    # right side, pad right
                    padr_x = stepx-(x2-x1)
                else:
                    # left side, pad left
                    padl_x = stepx-(x2-x1)
            # Horizontal lines
            y1 = setsy[j]
            y2 = setsy[j+1]

            if (y2-y1) > stepy:
                if j == 7:
                    y1 = y2 - stepy
                else:
                    y2 = y1 + stepy
            elif (y2-y1) < stepy:
                if j == 7:
                    # right side, pad right
                    padr_y = stepy-(y2-y1)
                else:
                    # left side, pad left
                    padl_y = stepy-(y2-y1)
            # slicing a, rows sliced with horizontal lines, cols by vertical lines so reversed
            # Also, change order so its A1,B1...H8 for a white-aligned board
            # Apply padding as defined previously to fit minor pixel offsets
            squares[:,:,(7-j)*8+i] = np.pad(a2[y1:y2, x1:x2],((padl_y,padr_y),(padl_x,padr_x)), mode="edge")
    return squares

def img2tiles(img):
    a = np.asarray(img.convert("L"), dtype=np.float32)
    A = tf.Variable(a)
    Dx = gradientx(A)
    Dy = gradienty(A)
    Dx_pos = tf.clip_by_value(Dx,    0.0, 255.0)
    Dx_neg = tf.clip_by_value(Dx, -255.0,   0.0)
    Dy_pos = tf.clip_by_value(Dy,    0.0, 255.0)
    Dy_neg = tf.clip_by_value(Dy, -255.0,   0.0)
    hough_Dx = tf.reduce_sum(Dx_pos, 0) * tf.reduce_sum(-Dx_neg, 0) / (a.shape[0]*a.shape[0])
    hough_Dy = tf.reduce_sum(Dy_pos, 1) * tf.reduce_sum(-Dy_neg, 1) / (a.shape[1]*a.shape[1])
    hough_Dx_thresh = tf.reduce_max(hough_Dx) * 3 / 5 * 0.9
    hough_Dy_thresh = tf.reduce_max(hough_Dy) * 3 / 5 * 0.9
    lines_x, lines_y, is_match = getChessLines(hough_Dx.numpy(), hough_Dy.numpy(), hough_Dx_thresh.numpy(), hough_Dy_thresh.numpy())
    # print("Chessboard found" if is_match else "Couldn"t find Chessboard")
    # print(f"X {lines_x} {np.diff(lines_x)}")
    # print(f"Y {lines_y} {np.diff(lines_y)}")
    tiles = np.empty([64, 32, 32])
    if is_match:
        tiles_unsized = getChessTiles(a, lines_x, lines_y)
        for i in range(64):
            tile = Image.fromarray(tiles_unsized[:,:,i])
            tile_resized = tile.resize([32, 32], Image.ADAPTIVE)
            tiles[i,:,:] = np.asarray(tile_resized)
    return is_match, tiles

if __name__ == "__main__":
    img_dirs = glob.glob("images/*.png")
    img = Image.open(img_dirs[0])
    is_match, tiles = img2tiles(img)
    print(is_match)
    print(tiles.shape)
    print(tiles)
