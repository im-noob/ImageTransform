import math
import cv2
import numpy as np


#helper functions

def getCubicNeighbors(x, size):#inter
    if x == 0:
        return [0, 0, 1, 2]

    if x == (size-1):
        return [-1, 0, -1, 0]

    if x == (size - 2):
        return [-1, 0, 1, 1]

    return [-1, 0, 1, 2] 

def getLanczos4Neighbors(x, size):#inter
    if x == 0:
        return [3, 2, 1, 0, 1, 2, 3, 4]  
    if x == 1:
        return [2, 1, -1, 0, 1, 2, 3, 4]
    if x == 2:
        return [1, -2, -1, 0, 1, 2, 3, 4]


    if x == size - 1:
        return [-3, -2, -1, 0, -1, -2, -3, 0]

    if x == size - 2:
        return [-3, -2, -1, 0, 1, -1, -2, -3]

    if x == size - 3:
        return [-3, -2, -1, 0, 1, 2, -1, -2]

    if x == size - 4:
        return [-3, -2, -1, 0, 1, 2, 3, 3]

    return [-3, -2, -1, 0, 1, 2, 3, 4] 

#1-d spline 
def spline(x, a):#inter
    x = abs(x)
    val = 0 
    if x >= 0 and x <= 1:
        return (a + 2)*(x ** 3) - (a + 3)*(x ** 2) + 1
    elif x > 1 and x <= 2:
        return a * (x ** 3) - (5 * a * x**2) + (8 * a * x) - (4 * a) 
    else:
        val = val

#1-d Lanczos, this technically works for any order but for our purposes it will always be 4
def lanczos(x, order):#inter
    x = abs(x)
    if x < order:
        if x == 0: #prevents divide by zero errors
            return 1
        else:
            return order * (math.sin( math.pi * x / order) * math.sin(math.pi * x)) / (math.pi ** 2 * x ** 2)
    else:
        return 0

#Get interpolation value for a lanczos-4 (64 pixel neighbors)
#The reason this has a optional order is because with some tweaks it can take any order
def lanczos4(image, x, y, order = 4):#inter
    x0 = math.floor(x)
    y0 = math.floor(y)
    
    xn = getLanczos4Neighbors(x0, image.shape[0])
    yn = getLanczos4Neighbors(y0, image.shape[1])

    val = 0

    for c in yn: 
        yv = 0
        for r in xn: 
            yv = yv + image[x0+r][y0+c] * lanczos(r, order)
        val = val + yv * lanczos(c, order) 
    
    return val



#Gets Interpolation value using 2-d splines
def bicubic(image, x, y, a = -.5):#inter
    x0 = math.floor(x) #row, height
    y0 = math.floor(y) #col, width

    xn = getCubicNeighbors(x0, image.shape[0])
    yn = getCubicNeighbors(y0, image.shape[1])
    val = 0
    
    for c in yn: #for every x value (every 1 neighbor)
        yv = 0
        for r in xn: #get y values first (get 4 y neighbors)
            yv = yv + image[x0+r][y0+c] * spline(r, a)
        val = val + yv * spline(c, a) #put together with x

    return val




#Functions below are the main functions to call 

def bicubic_interpolation(image, fx = 1, fy = 1, a = -.5):#type
    row, col, dim = image.shape
    nimg = np.zeros(shape=(int(row*fy), int(col*fx), dim), dtype=np.uint8)
    
    nr = int(row*fy)
    nc = int(col*fx)

    for x in range(nr):
        for y in range(nc):
            nimg[x][y] = bicubic(image, (x/fy), (y/fx))

    return nimg

#Specifically doesn't come with order as an argument because it will always be 4
def lanczos4_interpolation(image, fx = 1, fy = 1):#type
    row, col, dim = image.shape
    nimg = np.zeros(shape=(int(row*fy), int(col*fx), dim), dtype=np.uint8)
    
    nr = int(row*fy)
    nc = int(col*fx)

    for x in range(nr):
        for y in range(nc):
            nimg[x][y] = lanczos4(image, (x/fy), (y/fx))

    return nimg



# img = cv2.imread('per.jpeg')
# nimg = bicubic_interpolation(img, .5, .5)
# # nimg = lanczos4_interpolation(img, .5, .5)

# # cv2.imwrite('per.jpeg', nimg)
# ut.display_image(nimg)
