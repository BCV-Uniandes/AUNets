def right(rect, jit_x, jit_y):
  rect[0] += jit_x
  rect[2] += jit_x    
  return rect

def left(rect, jit_x, jit_y):
  rect[0] -= jit_x
  rect[2] -= jit_x    
  return rect
    
def down(rect, jit_x, jit_y):
  rect[1] += jit_y
  rect[3] += jit_y
  return rect
    
def up(rect, jit_x, jit_y):
  rect[1] -= jit_y
  rect[3] -= jit_y
  return rect 
       
def zoom_in(rect, jit_x, jit_y):
  rect[0] += jit_x
  rect[1] += jit_y
  rect[2] -= jit_x
  rect[3] -= jit_y
  return rect   
    
def zoom_out(rect, jit_x, jit_y):
  rect[0] -= jit_x
  rect[1] -= jit_y
  rect[2] += jit_x  
  rect[3] += jit_y
  return rect   
    
def rescale_rect(img, rect_img, new_img):
  #rect_img x1,y1,x2,y2
  import numpy as np
  imsz = np.array(img.shape[:2]).astype(np.float32)
  rect_norm = [rect_img[0]/imsz[1], rect_img[1]/imsz[0], rect_img[2]/imsz[1], rect_img[3]/imsz[0]]
  imsz_new = np.array(new_img.shape[:2]).astype(np.float32)
  new_rect = np.array([rect_norm[0]*imsz_new[1], rect_norm[1]*imsz_new[0], rect_norm[2]*imsz_new[1], rect_norm[3]*imsz_new[0]]).astype(np.uint16)
  return new_rect

def jitter_points(facePoints, max_values, mode):
  jit_x = int((facePoints[2]-facePoints[0])/8.0)#Third of width
  jit_y = int((facePoints[3]-facePoints[1])/8.0)#Third of height
  new_face = list(facePoints)
  #[x1,y1,x2,y2]
  if mode == 0:
    new_face = right(new_face, jit_x, jit_y)
  elif mode == 1:
    new_face = left(new_face, jit_x, jit_y)
  elif mode == 2:
    new_face = down(new_face, jit_x, jit_y)
  elif mode == 3:
    new_face = up(new_face, jit_x, jit_y)   
  elif mode == 4:
    new_face = right(new_face, jit_x, jit_y)
    new_face = down(new_face, jit_x, jit_y) 
  elif mode == 5:
    new_face = right(new_face, jit_x, jit_y)
    new_face = up(new_face, jit_x, jit_y)
  elif mode == 6:
    new_face = left(new_face, jit_x, jit_y)
    new_face = up(new_face, jit_x, jit_y)
  elif mode == 7:
    new_face = left(new_face, jit_x, jit_y)
    new_face = down(new_face, jit_x, jit_y)
  elif mode == 8:
    new_face = zoom_out(new_face, jit_x, jit_y)
  elif mode == 9:
    new_face = zoom_in(new_face, jit_x, jit_y)
  elif mode == 10:
    new_face = right(new_face, jit_x, jit_y)
    new_face = zoom_out(new_face, jit_x, jit_y)
  elif mode == 11:
    new_face = left(new_face, jit_x, jit_y)
    new_face = zoom_out(new_face, jit_x, jit_y)
  elif mode == 12:
    new_face = down(new_face, jit_x, jit_y)
    new_face = zoom_out(new_face, jit_x, jit_y)
  elif mode == 13:
    new_face = up(new_face, jit_x, jit_y)
    new_face = zoom_out(new_face, jit_x, jit_y)
  elif mode == 14:
    new_face = right(new_face, jit_x, jit_y)
    new_face = down(new_face, jit_x, jit_y) 
    new_face = zoom_out(new_face, jit_x, jit_y)
  elif mode == 15:
    new_face = right(new_face, jit_x, jit_y)
    new_face = up(new_face, jit_x, jit_y)
    new_face = zoom_out(new_face, jit_x, jit_y)
  elif mode == 16:
    new_face = left(new_face, jit_x, jit_y)
    new_face = up(new_face, jit_x, jit_y)
    new_face = zoom_out(new_face, jit_x, jit_y)
  elif mode == 17:
    new_face = left(new_face, jit_x, jit_y)
    new_face = down(new_face, jit_x, jit_y)
    new_face = zoom_out(new_face, jit_x, jit_y)
  elif mode == 18:
    new_face = right(new_face, jit_x, jit_y)
    new_face = zoom_in(new_face, jit_x, jit_y)
  elif mode == 19:
    new_face = left(new_face, jit_x, jit_y)
    new_face = zoom_in(new_face, jit_x, jit_y)
  elif mode == 20:
    new_face = down(new_face, jit_x, jit_y)
    new_face = zoom_in(new_face, jit_x, jit_y)
  elif mode == 21:
    new_face = up(new_face, jit_x, jit_y)
    new_face = zoom_in(new_face, jit_x, jit_y)
  elif mode == 22:
    new_face = right(new_face, jit_x, jit_y)
    new_face = down(new_face, jit_x, jit_y) 
    new_face = zoom_in(new_face, jit_x, jit_y)
  elif mode == 23:
    new_face = right(new_face, jit_x, jit_y)
    new_face = up(new_face, jit_x, jit_y)
    new_face = zoom_in(new_face, jit_x, jit_y)
  elif mode == 24:
    new_face = left(new_face, jit_x, jit_y)
    new_face = up(new_face, jit_x, jit_y)
    new_face = zoom_in(new_face, jit_x, jit_y)
  elif mode == 25:
    new_face = left(new_face, jit_x, jit_y)
    new_face = down(new_face, jit_x, jit_y)
    new_face = zoom_in(new_face, jit_x, jit_y)
  elif mode == 26:
    new_face = right(new_face, jit_x*2, jit_y)
  elif mode == 27:
    new_face = left(new_face, jit_x*2, jit_y)
  elif mode == 28:
    new_face = down(new_face, jit_x*2, jit_y)
  elif mode == 29:
    new_face = up(new_face, jit_x*2, jit_y)   
  elif mode == 30:
    new_face = right(new_face, jit_x*2, jit_y)
    new_face = down(new_face, jit_x*2, jit_y) 
  elif mode == 31:
    new_face = right(new_face, jit_x*2, jit_y)
    new_face = up(new_face, jit_x*2, jit_y)
  elif mode == 32:
    new_face = left(new_face, jit_x*2, jit_y)
    new_face = up(new_face, jit_x*2, jit_y)
  elif mode == 33:
    new_face = left(new_face, jit_x*2, jit_y)
    new_face = down(new_face, jit_x*2, jit_y)
  elif mode == 34:
    new_face = zoom_out(new_face, jit_x*2, jit_y)
  elif mode == 35:
    new_face = zoom_in(new_face, jit_x*2, jit_y)
  elif mode == 36:
    new_face = right(new_face, jit_x*2, jit_y)
    new_face = zoom_out(new_face, jit_x*2, jit_y)
  elif mode == 37:
    new_face = left(new_face, jit_x*2, jit_y)
    new_face = zoom_out(new_face, jit_x*2, jit_y)
  elif mode == 38:
    new_face = down(new_face, jit_x*2, jit_y)
    new_face = zoom_out(new_face, jit_x*2, jit_y)
  elif mode == 39:
    new_face = up(new_face, jit_x*2, jit_y)
    new_face = zoom_out(new_face, jit_x*2, jit_y)
  elif mode == 40:
    new_face = right(new_face, jit_x*2, jit_y)
    new_face = down(new_face, jit_x*2, jit_y) 
    new_face = zoom_out(new_face, jit_x*2, jit_y)
  elif mode == 41:
    new_face = right(new_face, jit_x*2, jit_y)
    new_face = up(new_face, jit_x*2, jit_y)
    new_face = zoom_out(new_face, jit_x*2, jit_y)
  elif mode == 42:
    new_face = left(new_face, jit_x*2, jit_y)
    new_face = up(new_face, jit_x*2, jit_y)
    new_face = zoom_out(new_face, jit_x*2, jit_y)
  elif mode == 43:
    new_face = left(new_face, jit_x*2, jit_y)
    new_face = down(new_face, jit_x*2, jit_y)
    new_face = zoom_out(new_face, jit_x*2, jit_y)
  elif mode == 44:
    new_face = right(new_face, jit_x*2, jit_y)
    new_face = zoom_in(new_face, jit_x*2, jit_y)
  elif mode == 45:
    new_face = left(new_face, jit_x*2, jit_y)
    new_face = zoom_in(new_face, jit_x*2, jit_y)
  elif mode == 46:
    new_face = down(new_face, jit_x*2, jit_y)
    new_face = zoom_in(new_face, jit_x*2, jit_y)
  elif mode == 47:
    new_face = up(new_face, jit_x*2, jit_y)
    new_face = zoom_in(new_face, jit_x*2, jit_y)
  elif mode == 48:
    new_face = right(new_face, jit_x*2, jit_y)
    new_face = down(new_face, jit_x*2, jit_y) 
    new_face = zoom_in(new_face, jit_x*2, jit_y)
  elif mode == 49:
    new_face = right(new_face, jit_x*2, jit_y)
    new_face = up(new_face, jit_x*2, jit_y)
    new_face = zoom_in(new_face, jit_x*2, jit_y)
  elif mode == 50:
    new_face = left(new_face, jit_x*2, jit_y)
    new_face = up(new_face, jit_x*2, jit_y)
    new_face = zoom_in(new_face, jit_x*2, jit_y)
  elif mode == 51:
    new_face = left(new_face, jit_x*2, jit_y)
    new_face = down(new_face, jit_x*2, jit_y)
    new_face = zoom_in(new_face, jit_x*2, jit_y)
  elif mode == 52:
    new_face = right(new_face, jit_x, jit_y*2)    
  elif mode == 53:
    new_face = left(new_face, jit_x, jit_y*2)
  elif mode == 54:
    new_face = down(new_face, jit_x, jit_y*2)
  elif mode == 55:
    new_face = up(new_face, jit_x, jit_y*2)   
  elif mode == 56:
    new_face = right(new_face, jit_x, jit_y*2)
    new_face = down(new_face, jit_x, jit_y*2) 
  elif mode == 57:
    new_face = right(new_face, jit_x, jit_y*2)
    new_face = up(new_face, jit_x, jit_y*2)
  elif mode == 58:
    new_face = left(new_face, jit_x, jit_y*2)
    new_face = up(new_face, jit_x, jit_y*2)
  elif mode == 59:
    new_face = left(new_face, jit_x, jit_y*2)
    new_face = down(new_face, jit_x, jit_y*2)
  elif mode == 60:
    new_face = zoom_out(new_face, jit_x, jit_y*2)
  elif mode == 61:
    new_face = zoom_in(new_face, jit_x, jit_y*2)
  elif mode == 62:
    new_face = right(new_face, jit_x, jit_y*2)
    new_face = zoom_out(new_face, jit_x, jit_y*2)
  elif mode == 63:
    new_face = left(new_face, jit_x, jit_y*2)
    new_face = zoom_out(new_face, jit_x, jit_y*2)
  elif mode == 64:
    new_face = down(new_face, jit_x, jit_y*2)
    new_face = zoom_out(new_face, jit_x, jit_y*2)
  elif mode == 65:
    new_face = up(new_face, jit_x, jit_y*2)
    new_face = zoom_out(new_face, jit_x, jit_y*2)
  elif mode == 66:
    new_face = right(new_face, jit_x, jit_y*2)
    new_face = down(new_face, jit_x, jit_y*2) 
    new_face = zoom_out(new_face, jit_x, jit_y*2)
  elif mode == 67:
    new_face = right(new_face, jit_x, jit_y*2)
    new_face = up(new_face, jit_x, jit_y*2)
    new_face = zoom_out(new_face, jit_x, jit_y*2)
  elif mode == 68:
    new_face = left(new_face, jit_x, jit_y*2)
    new_face = up(new_face, jit_x, jit_y*2)
    new_face = zoom_out(new_face, jit_x, jit_y*2)
  elif mode == 69:
    new_face = left(new_face, jit_x, jit_y*2)
    new_face = down(new_face, jit_x, jit_y*2)
    new_face = zoom_out(new_face, jit_x, jit_y*2)
  elif mode == 70:
    new_face = right(new_face, jit_x, jit_y*2)
    new_face = zoom_in(new_face, jit_x, jit_y*2)
  elif mode == 71:
    new_face = left(new_face, jit_x, jit_y*2)
    new_face = zoom_in(new_face, jit_x, jit_y*2)
  elif mode == 72:
    new_face = down(new_face, jit_x, jit_y*2)
    new_face = zoom_in(new_face, jit_x, jit_y*2)
  elif mode == 72:
    new_face = up(new_face, jit_x, jit_y*2)
    new_face = zoom_in(new_face, jit_x, jit_y*2)
  elif mode == 73:
    new_face = right(new_face, jit_x, jit_y*2)
    new_face = down(new_face, jit_x, jit_y*2) 
    new_face = zoom_in(new_face, jit_x, jit_y*2)
  elif mode == 74:
    new_face = right(new_face, jit_x, jit_y*2)
    new_face = up(new_face, jit_x, jit_y*2)
    new_face = zoom_in(new_face, jit_x, jit_y*2)
  elif mode == 75:
    new_face = left(new_face, jit_x, jit_y*2)
    new_face = up(new_face, jit_x, jit_y*2)
    new_face = zoom_in(new_face, jit_x, jit_y*2)
  elif mode == 76:
    new_face = left(new_face, jit_x, jit_y*2)
    new_face = down(new_face, jit_x, jit_y*2)
    new_face = zoom_in(new_face, jit_x, jit_y*2)      

  new_face = [int(max(new_face[0],0)), int(max(new_face[1],0)), int(min(new_face[2], max_values[1])), int(min(new_face[3], max_values[0]))]    
  return new_face
