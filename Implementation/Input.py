

class Input:
    method = None #string
    image = None #image matrix

    interpolation = None #string, used for scaling
    fx = None #float, used for scaling
    fy = None #float, used for scaling
    x = None #float, used for rotation, translation, perpective
    y = None #float, used for rotation, translation, perpective
    z = None #float, used for perpective
    r = None #float, used for rotation, perpective
    a11 = None
    a12 = None
    a21 = None
    a22 = None
    r = None
    r_value = None

    def printAll(self):
        """initilize everything before calling"""
        # print('method: ' + method)
        print('image: ' + image)
        print('interpolation: ' + interpolation)
        print('fx: ' + fx)
        print('fy: ' + fy)
        print('x: ' + x)
        print('y: ' + y)
        print('z: ' + z)
        print('r: ' + r)

