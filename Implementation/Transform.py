import cv2
from Implementation import Utility as ut
from Implementation import Input
import numpy as np
import math


class Transform:

    def trans(self, input):
        """the only function the GUI calls
        performs transformation based on input variables
        returns resulting image"""

        method = input.method
        image = input.image

        result = None

        print("for recheck")
        print(input.x)
        print(input.y)
        print(input.z)
        print(input.fx)
        print(input.fy)
        print(input.r_value)
        print(input.method)
        print(input.interpolation)

        # i have changed r to r_value due to some issue with r

        if method == "test":
            result = self.test(image)
        elif method == "scale":
            result = self.scale(input.fx, input.fy, input.interpolation, image)
        elif method == "rotate":
            result = self.rotate(input.x, input.y, input.r_value, image)
        elif method == "translate":
            result = self.translate(input.x, input.y, image)
        elif method == "affine":
            result = self.affineTransform(input.a11, input.a12, input.a21, input.a22, input.x, input.y, image)
        elif method == "shear":
            result = self.shear(input.x, input.y, image)
        elif method == "perspective":
            result = self.perspective(input.x, input.y, input.z, input.r_value, image)
        elif method == "polar":
            result = self.polar(image)
        elif method == "logpolar":
            result = self.logpolar(image)
        else:
            print("Error: @trams: unidentified method name.")

        return result

    def scale(self, fx, fy, interpolation, image):#inter
        result = None
        if (interpolation == "neighbor"):
            result = self.nearest_neighbor(fx, fy, image)
        elif (interpolation == "bilinear"):
            result = self.bilinear(fx, fy, image)
        elif (interpolation == "bicubic"):
            result = self.bicubic_interpolation(image, fx, fy, -.5)
        elif (interpolation == "lanczos4"):
            result = self.lanczos4_interpolation(image, fx, fy)
        else:
            print("Error: @scale: unidentified interpolation name.")

        return result

    """~~~~~~~~~~~~~~~~~~~~~~interpolations~~~~~~~~~~~~~~~~~~~~~~"""

    def nearest_neighbor(self, fx, fy, image):#inter
        i = image.shape[0]
        j = image.shape[1]
        resampled = np.zeros((round(i * fy), round(j * fx), 3), image.dtype)

        for r in range(resampled.shape[0]):
            for c in range(resampled.shape[1]):
                index_row = math.floor(r / fy)
                index_col = math.floor(c / fx)
                resampled[r, c] = image[index_row, index_col]

        return resampled

    def bilinear(self, fx, fy, image):#inter
        i = image.shape[0]
        j = image.shape[1]
        resampled = np.zeros((round(i * fy), round(j * fx), 3), image.dtype)
        for r in range(resampled.shape[0]):
            for c in range(resampled.shape[1]):
                point = [0] * 3
                point[0] = c / fx
                point[1] = r / fy
                rr = math.floor(r / fy)
                cc = math.floor(c / fx)
                if (rr == i - 1):
                    rr -= 1
                if (cc == j - 1):
                    cc -= 1
                self.bilinear_interpolation([cc, rr, image[rr][cc]], [cc + 1, rr, image[rr][cc + 1]],
                                          [cc, rr + 1, image[rr + 1][cc]], [cc + 1, rr + 1, image[rr + 1][cc + 1]],
                                          point)
                resampled[r][c] = point[2]

        return resampled

    def linear_interpolation(self, pt1, pt2, unknown, alongX = 1):#inter
        if(alongX):
            return (pt1[2] + (((unknown[0] - pt1[0])) * ((int(pt2[2]) - int(pt1[2])))))
        else :
            return pt1[2] + (unknown[1] - pt1[1]) * (int(pt2[2]) - int(pt1[2]))

    def bilinear_interpolation(self, pt1, pt2, pt3, pt4, unknown):#inter
        v1 = self.linear_interpolation(pt1,pt2,unknown,1)
        v2 = self.linear_interpolation(pt3, pt4, unknown, 1)
        unknown[2] = self.linear_interpolation([0,pt1[1],v1],[0,pt3[1],v2],unknown,0)
        return unknown

    """
        Keeps values in between 0 and 255
    """
    def clamp(self, image, val):#inter
        if len(image.shape) > 2:
            for x, y in zip(val, range(image.shape[2])) : 
                if x < 0: 
                    val[y] = 0 

                if x > 255:
                    val[y] = 255
            return val
        else:
            if val < 0:
                val = 0
            if val > 255:
                val = 255
            return val

    """
        Checks for out of bounds, assumes the image is at least 4 x 4,
        it does not check for, or account for images less than 4 x 4
    """

    def getCubicNeighbors(self, x, size):#inter
        if x == 0:
            return [0, 0, 1, 2]

        if x == (size-1):
            return [-1, 0, -1, 0]

        if x == (size - 2):
            return [-1, 0, 1, 1]

        return [-1, 0, 1, 2] 

    
    """
        Calculates the variable responsible for avereging the values
    """
    def spline(self, x, a):#inter
        x = abs(x)
        val = 0 
        if x >= 0 and x <= 1:
            return (a + 2)*(x ** 3) - (a + 3)*(x ** 2) + 1
        elif x > 1 and x <= 2:
            return a * (x ** 3) - (5 * a * x**2) + (8 * a * x) - (4 * a) 
        else:
            val = val

    
    """
        Calculates the interpolation of 1 pixel using splines
    """
    def bicubic(self, image, x, y, a = -.5):#inter
        x0 = math.floor(x) #row, height
        y0 = math.floor(y) #col, width

        xn = self.getCubicNeighbors(x0, image.shape[0])
        yn = self.getCubicNeighbors(y0, image.shape[1])
        val = 0
        
        for c in yn: #for every x value (every 1 neighbor)
            yv = 0
            for r in xn: #get y values first (get 4 y neighbors)
                yv = yv + image[x0+r][y0+c] * self.spline(x - (x0+r), a)
            val = val + yv * self.spline(y - (y0+c), a) #put together with x
    
        val = self.clamp(image, val)
        return val

    
    """
        Bicubic Interpolation Main Function
        Inputs  : image, x-scale, y-scale, shaprness 
                  (Note on sharpness, can be a user value but it must be negative
                  in between -1 and 0, it is best left with its default value at -.5)
        Assumes : image is at least 4 x 4
        Output  : Interpolated image, does not effect the original
    """
    def bicubic_interpolation(self, image, fx = 1, fy = 1, a = -.5):#inter
        row, col, dim = image.shape
        nimg = np.zeros(shape=(int(row*fy), int(col*fx), dim), dtype=np.uint8)
        
        nr = int(row*fy)
        nc = int(col*fx)

        for x in range(nr):
            for y in range(nc):
                nimg[x][y] = self.bicubic(image, (x/fy), (y/fx))

        return nimg

    """
        Checks for out of bounds, assumes the image is at least 8 x 8,
        it does not check for, or account for images less than 8 x 8
    """
    def getLanczos4Neighbors(self, x, size):#inter
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
    
    """
        Calculates the variable responsible for avereging the values
        Works for any order i.e. lanczos-3, 10, 2 ...
    """
    def lanczos(self, x, order):#inter
        x = abs(x)
        if x < order:
            if x == 0: #prevents divide by zero errors
                return 1
            else:
                return order * (math.sin( math.pi * x / order) * math.sin(math.pi * x)) / (math.pi ** 2 * x ** 2)
        else:
            return 0

    """
        Calculates the interpolation of 1 pixel using lanczos calculations.
        Can be modified to work for any order i.e. lanzcos-3, 2, 10 by changing
        out the neighbor functions to the corresponding about. 
    """
    def lanczos4(self, image, x, y, order = 4):#inter
        x0 = math.floor(x)
        y0 = math.floor(y)
        
        xn = self.getLanczos4Neighbors(x0, image.shape[0])
        yn = self.getLanczos4Neighbors(y0, image.shape[1])

        val = 0

        for c in yn: 
            yv = 0
            for r in xn: 
                yv = yv + image[x0+r][y0+c] * self.lanczos(x - (x0+r), order)
            val = val + yv * self.lanczos(y - (y0+c), order)
        
        val = self.clamp(image, val)
        return val

    """
        Lanczos-4 Interpolation Main Function
        Inputs  : image, x-scale, y-scale 
        Assumes : image is at least 8 x 8
        Output  : Interpolated image, does not effect the original
    """
    def lanczos4_interpolation(self, image, fx = 1, fy = 1):#inter
        row, col, dim = image.shape
        nimg = np.zeros(shape=(int(row*fy), int(col*fx), dim), dtype=np.uint8)
        
        nr = int(row*fy)
        nc = int(col*fx)

        for x in range(nr):
            for y in range(nc):
                nimg[x][y] = self.lanczos4(image, (x/fy), (y/fx))

        return nimg



    """~~~~~~~~~~~~~~~~~~~~~~end of interpolations~~~~~~~~~~~~~~~~~~~~~~"""

    def rotate(self, x, y, r, image):#inter

        return image

    def translate(self, x, y, image):#inter
        result = np.zeros((800, 800, 3), np.uint8)
        for ix in range(0, image.shape[1]):
            for iy in range(0, image.shape[0]):
                x1 = ix + x
                y1 = iy + y
                if(x1 < 800 and x1 >= 0 and y1 < 800 and y1 >= 0):
                   result[y1][x1] = image[iy][ix]
        return result

    def shear(self, x_shear, y_shear, image):#inter
        rx_shear = 2
        y_shear = 2
        result = np.zeros((800, 800, 3), np.uint8)
        for ix in range(0, image.shape[1]):
            for iy in range(0, image.shape[0]):
                fx = ix + iy * x_shear
                fy = iy + ix * y_shear
                if fx >= 0 and fx < 800 and fy >= 0 and fy < 800:
                   print(0)
                   result[fy][fx] = image[iy][ix]

        return result

    def affineTransform(self, a11,a12,a21,a22, x, y,image):#inter
        """i dunno what parameters are needed here ;("""
        result = np.zeros((800, 800, 3), np.uint8)
        for ix in range(0, image.shape[1]):
            for iy in range(0, image.shape[0]):
                fx = math.floor(ix * a11 + iy * a12) + x
                fy = math.floor(ix * a21 + iy * a22) + y
                if fx >= 0 and fx < 800 and fy >= 0 and fy < 800:
                   result[fy][fx] = image[iy][ix]

        return result

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    def perspective(self, x, y, z, r, image):#inter
        r = -r
        d = 100
        result = np.zeros((800, 800, 3), np.uint8)
        m = math.tan(r)

        ip_x = x + math.cos(r) * d
        ip_y = y + math.sin(r) * d

        for iy in range(0, image.shape[0]):
            for ix in range(0, image.shape[1]):
               isInX = False
               if ix != x :
                   lm = (iy - y) / (ix - x)
                   if(m != 0) :
                       fx = (y - ip_x/m - ip_y - lm*x) / (-lm - 1/m)
                       fy = -1/m * fx + ip_x / m + ip_y
                   else :
                       fx = x + d
                       fy = lm * fx - lm*x + y
               else:
                   if(m != 0) :
                       fy = iy
                       fx = (-fy + ip_x / m + ip_y) * m
                   else :
                       fx = x + d
                       fy = iy


               isInY = False
               if (m != 0 and y != iy):
                   isInY = True
                   #c = ip_x / m + ip_y
                   hm = z / (y - iy)
                   hy = hm * fy - hm * y + z

               if (math.fabs(m) < 1e10 and x != ix):
                   isInX = True
                   #c = - ip_x - ip_y * m
                   hm = z / (x - ix)
                   hx = hm * fx - hm * x + z

               dot = (x - ix)*math.cos(r) + (y - iy)*math.sin(r)
               if (dot > 0) : continue
               fz = 0
               if isInY :
                   fz = hy
               elif isInX :
                   fz = hx

               fz = fz - z + 400
               distance = math.sqrt((ip_x - fx) ** 2 + (ip_y - fy) ** 2)
               if (distance < 400 and fz < 800 and fz >= 0):
                    cpv = (ip_x - x)*(fy - y) - (ip_y - y)*(fx - x)
                    if cpv > 0 : sign = 1
                    else : sign = -1
                    ff = sign * distance + 400
                    result[799 - math.floor(fz)][math.floor(ff)] = image[iy][ix]
        return result

    def polar(self, image):#inter
        #smooth pic ratio: x:1 y:6
        buff = np.zeros((image.shape[1],image.shape[1],image.shape[2]), np.uint8)
        radius = math.floor(image.shape[1] / 2)
        result = np.zeros((image.shape[1], image.shape[1], image.shape[2]), np.uint8)

        for r in range (0, image.shape[1]):
            for d in range (0, image.shape[0]):
                delta = d * (-math.pi * 2) / image.shape[0]
                y = math.floor(radius + math.floor(r/2) * math.sin(delta))
                x = math.floor(radius + math.floor(r/2) * math.cos(delta))
                buff[y][x][0] = 1
                result[y][x] = image[d][r]

        return self.polar_post_processing(result,buff,radius)

    def polar_post_processing(self, result, buff, radius):#inter
        temp = np.zeros((result.shape[1], result.shape[1], result.shape[2]), np.uint8)
        for x in range (0, result.shape[1]):
            for y in range (0, result.shape[0]):
                d = math.sqrt((x - result.shape[1] / 2)*(x - result.shape[1] / 2) + (y - result.shape[0] / 2)*(y - result.shape[0] / 2))
                if d <= radius :
                    u = y-1
                    d = y+1
                    l = x-1
                    r = x+1

                    uu = u - 1
                    dd = d + 1
                    ll = l + 1
                    rr = r + 1

                    if u < 0 : u = 0
                    if d >= result.shape[0] : d = result.shape[0] - 1
                    if l < 0 : l = 0
                    if r >= result.shape[1] : r = result.shape[1] - 1
                    if uu < 0 : uu = 0
                    if dd >= result.shape[0] : dd = result.shape[0] - 1
                    if ll < 0 : ll = 0
                    if rr >= result.shape[1] : rr = result.shape[1] - 1

                    if buff[u][x][0] == 1 :
                        temp[y][x] = result[u][x]
                    elif buff[d][x][0] == 1 :
                        temp[y][x] = result[d][x]
                    elif buff[y][l][0] == 1 :
                        temp[y][x] = result[y][l]
                    elif buff[y][r][0] == 1 :
                        temp[y][x] = result[y][r]

                    elif buff[uu][x][0] == 1 :
                        temp[y][x] = result[uu][x]
                    elif buff[dd][x][0] == 1:
                        temp[y][x] = result[dd][x]
                    elif buff[y][ll][0] == 1:
                        temp[y][x] = result[y][ll]
                    elif buff[y][rr][0] == 1:
                        temp[y][x] = result[y][rr]


        return temp

    def logpolar(self, image):#inter
        b = 2
        buff = np.zeros((image.shape[1], image.shape[1], image.shape[2]), np.uint8)
        radius = math.floor(image.shape[1] / 2)
        result = np.zeros((image.shape[1], image.shape[1], image.shape[2]), np.uint8)
        mx = math.log(image.shape[1] - 1, b)
        mr = math.floor( image.shape[1] / mx / 2)

        for r in range(0, image.shape[1]):
            for d in range(0, image.shape[0]):
                delta = d * (-math.pi * 2) / image.shape[0]
                lr = 0
                if r > 0 : lr = math.log( image.shape[1] - r, b)
                y = math.floor(radius + math.floor((mx - lr) * mr) * math.sin(delta))
                x = math.floor(radius + math.floor((mx - lr) * mr) * math.cos(delta))
                buff[y][x][0] = 1
                result[y][x] = image[d][r]
        return result

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    def test(self, image):#inter

        return image
