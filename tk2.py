#using python 3
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
from transforms import RGBTransform # from source code mentioned above


from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

''' updating code '''
from Implementation import Utility as ut
from Implementation import Input
from Implementation import Transform
import math


root = Tk()


# frame = tk.Frame(root)
# root.attributes('-fullscreen', True)

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

negative_btn_fire_status = 0 
log_btn_fire_status = 0
power_btn_fire_status = 0
histogran_btn_fire_status = 0
log_btn_label = Label()
power_btn_label = Label()
negative_btn_label = Label()
histogran_btn_label = Label()
#methods

def negative_btn_fire(event):

	global image
	global main_img
	global filename
	global negative_btn_label
	global log_btn_label
	global power_btn_label
	global histogran_btn_label
	global negative_btn_fire_status
	global log_btn_fire_status
	global power_btn_fire_status
	global histogran_btn_fire_status
	if(negative_btn_fire_status == 0):
		#remocin other label ands tatus 
		negative_btn_label.pack_forget()
		negative_btn_fire_status = 0
		log_btn_label.pack_forget()
		log_btn_fire_status = 0
		power_btn_label.pack_forget()
		power_btn_fire_status = 0
		histogran_btn_label.pack_forget()
		histogran_btn_fire_status = 0

		print("negative_btn_fire")
		negative_btn_label = Label(pw5,text = 'Negative')
		negative_btn_label.pack(padx=50, side=LEFT)

		negative_btn_fire_status = 1
		#changing image main image
		main_img.pack_forget()
		cv_img = cv2.imread(filename)
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)		#correction cv2 problem 
		cv_img = cv2.bitwise_not(cv_img)						#converting to negative
		cv_img =   cv2.resize(cv_img, (int(screen_height*(0.75)), int(screen_height*(0.75)) ))
		img = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
		main_img = tk.Label(pw1,image=img, height=screen_height*(0.75), width =screen_width*(0.75))
		main_img.image = img 
		main_img.pack(side = "bottom", fill = "both", expand = "yes")
	else:
		print("already added")
		negative_btn_label.pack_forget()
		negative_btn_fire_status = 0
		#changing image main image
		main_img.pack_forget()
		cv_img = cv2.imread(filename)
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)		#correction cv2 problem 
		cv_img =   cv2.resize(cv_img, (int(screen_height*(0.75)), int(screen_height*(0.75)) ))
		img = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
		main_img = tk.Label(pw1,image=img, height=screen_height*(0.75), width =screen_width*(0.75))
		main_img.image = img 
		main_img.pack(side = "bottom", fill = "both", expand = "yes")

#log transform 
def log_trans(image):
	image = np.uint8(np.log1p(image))
	thresh = 1
	image = cv2.threshold(image,thresh,255,cv2.THRESH_BINARY)[1]
	return image

#power law tranmissiotn 
def adjust_gamma(image, gamma=10):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def hist_trans(image):
	hist = cv2.calcHist([image],[0],None,[256],[0,256])
	plt.hist(image.ravel(),256,[0,256]);
	plt.show()
	return hist

#for rbg transformation
def rgb_trans(R,G,B):
	global image
	global main_img
	global filename
	print("RBG trans")
	lena = Image.open(filename)
	lena = lena.convert('RGB') # ensure image has 3 channels
	color_img = RGBTransform().mix_with((R, G, B),factor=.30).applied_to(lena) 
	

	rgb_img =  cv2.cvtColor(np.array(color_img), cv2.COLOR_RGB2BGR)		#PIL to cv2

	#updating iamge
	main_img.pack_forget()
	cv_img = rgb_img					#storing to rbg filter
	cv_img =   cv2.resize(cv_img, (int(screen_height*(0.75)), int(screen_height*(0.75)) ))
	img = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
	main_img = tk.Label(pw1,image=img, height=screen_height*(0.75), width =screen_width*(0.75))
	main_img.image = img 
	main_img.pack(side = "bottom", fill = "both", expand = "yes")
	return

def apply_rbg(event):
	# global T_x
	R_s = T_R.get('1.0', 'end-1c')
	G_s = T_G.get('1.0', 'end-1c')
	B_s = T_B.get('1.0', 'end-1c')
	if(len(R_s) == 0):
		R = 0
	else:
		R = int(R_s)

	if(len(G_s) == 0):
		G = 0
	else:
		G = int(G_s)

	if(len(B_s) == 0):
		B = 0
	else:
		B = int(B_s)

	print(R)
	print(G)
	print(B)

	rgb_trans(R,G,B)


def setTextRBG():
	T_G.insert('1.0', '0')
	T_B.insert('1.0', '0')
	T_R.insert('1.0', '0')

def log_btn_fire(event):
	global image
	global main_img
	global filename
	global negative_btn_label
	global log_btn_label
	global power_btn_label
	global histogran_btn_label
	global negative_btn_fire_status
	global log_btn_fire_status
	global power_btn_fire_status
	global histogran_btn_fire_status
	if(log_btn_fire_status == 0):

		#remocin other label ands tatus 
		negative_btn_label.pack_forget()
		negative_btn_fire_status = 0
		log_btn_label.pack_forget()
		log_btn_fire_status = 0
		power_btn_label.pack_forget()
		power_btn_fire_status = 0
		histogran_btn_label.pack_forget()
		histogran_btn_fire_status = 0

		log_btn_label = Label(pw5,text = 'Log Transformation')
		log_btn_label.pack(padx=50, side=LEFT)
		log_btn_fire_status = 1

		#changing image main image
		main_img.pack_forget()
		cv_img = cv2.imread(filename)
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)		#correction cv2 problem 

		#converting to negative
		cv_img = log_trans(cv_img)
				
		
		cv_img =   cv2.resize(cv_img, (int(screen_height*(0.75)), int(screen_height*(0.75)) ))
		img = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
		main_img = tk.Label(pw1,image=img, height=screen_height*(0.75), width =screen_width*(0.75))
		main_img.image = img 
		main_img.pack(side = "bottom", fill = "both", expand = "yes")

	else:
		log_btn_label.pack_forget()
		log_btn_fire_status = 0

		#changing image main image
		main_img.pack_forget()
		cv_img = cv2.imread(filename)
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)		#correction cv2 problem 
		cv_img =   cv2.resize(cv_img, (int(screen_height*(0.75)), int(screen_height*(0.75)) ))
		img = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
		main_img = tk.Label(pw1,image=img, height=screen_height*(0.75), width =screen_width*(0.75))
		main_img.image = img 
		main_img.pack(side = "bottom", fill = "both", expand = "yes")

def power_btn_fire(event):
	global image
	global main_img
	global filename
	global negative_btn_label
	global log_btn_label
	global power_btn_label
	global histogran_btn_label
	global negative_btn_fire_status
	global log_btn_fire_status
	global power_btn_fire_status
	global histogran_btn_fire_status
	if(power_btn_fire_status == 0):

		#remocin other label ands tatus 
		negative_btn_label.pack_forget()
		negative_btn_fire_status = 0
		log_btn_label.pack_forget()
		log_btn_fire_status = 0
		power_btn_label.pack_forget()
		power_btn_fire_status = 0
		histogran_btn_label.pack_forget()
		histogran_btn_fire_status = 0

		power_btn_label = Label(pw5,text = 'Power Law Transformation')
		power_btn_label.pack(padx=50, side=LEFT)
		power_btn_fire_status = 1

		#changing image main image
		main_img.pack_forget()
		cv_img = cv2.imread(filename)
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)		#correction cv2 problem 
		cv_img = adjust_gamma(cv_img)						#converting to  gamma adjust for gama value 10 
		cv_img =   cv2.resize(cv_img, (int(screen_height*(0.75)), int(screen_height*(0.75)) ))
		img = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
		main_img = tk.Label(pw1,image=img, height=screen_height*(0.75), width =screen_width*(0.75))
		main_img.image = img 
		main_img.pack(side = "bottom", fill = "both", expand = "yes")

	else:
		power_btn_label.pack_forget()
		power_btn_fire_status = 0

		#changing image main image
		main_img.pack_forget()
		cv_img = cv2.imread(filename)
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)		#correction cv2 problem 
		cv_img =   cv2.resize(cv_img, (int(screen_height*(0.75)), int(screen_height*(0.75)) ))
		img = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
		main_img = tk.Label(pw1,image=img, height=screen_height*(0.75), width =screen_width*(0.75))
		main_img.image = img 
		main_img.pack(side = "bottom", fill = "both", expand = "yes")

def histogran_btn_fire(event):
	global image
	global main_img
	global filename
	global negative_btn_label
	global log_btn_label
	global power_btn_label
	global histogran_btn_label
	global negative_btn_fire_status
	global log_btn_fire_status
	global power_btn_fire_status
	global histogran_btn_fire_status
	if(histogran_btn_fire_status == 0):

		#remocin other label ands tatus 
		negative_btn_label.pack_forget()
		negative_btn_fire_status = 0
		log_btn_label.pack_forget()
		log_btn_fire_status = 0
		power_btn_label.pack_forget()
		power_btn_fire_status = 0
		histogran_btn_label.pack_forget()
		histogran_btn_fire_status = 0

		histogran_btn_label = Label(pw5,text = 'Histogran')
		histogran_btn_label.pack(padx=50, side=LEFT)
		histogran_btn_fire_status = 1

		#changing image main image
		main_img.pack_forget()
		cv_img = cv2.imread(filename)
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)		#correction cv2 problem 
		cv_img = hist_trans(cv2.imread(filename,0))	#converting to histogram
		cv_img =   cv2.resize(cv_img, (int(screen_height*(0.75)), int(screen_height*(0.75)) ))
		img = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
		main_img = tk.Label(pw1,image=img, height=screen_height*(0.75), width =screen_width*(0.75))
		main_img.image = img 
		main_img.pack(side = "bottom", fill = "both", expand = "yes")

	else:
		histogran_btn_label.pack_forget()
		histogran_btn_fire_status = 0

		#changing image main image
		main_img.pack_forget()
		cv_img = cv2.imread(filename)
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)		#correction cv2 problem 
		cv_img =   cv2.resize(cv_img, (int(screen_height*(0.75)), int(screen_height*(0.75)) ))
		img = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
		main_img = tk.Label(pw1,image=img, height=screen_height*(0.75), width =screen_width*(0.75))
		main_img.image = img 
		main_img.pack(side = "bottom", fill = "both", expand = "yes")
	
filename = "per.jpeg"
def getfile():
	global image
	global main_img
	global org_picture
	global pw3
	global filename
	global negative_btn_label
	global log_btn_label
	global power_btn_label
	global histogran_btn_label
	global negative_btn_fire_status
	global log_btn_fire_status
	global power_btn_fire_status
	global histogran_btn_fire_status
	filename =  filedialog.askopenfilename (initialdir = "/",title = "Select file")
	filename_copy = filename
	#changing image main image
	main_img.pack_forget()
	image = Image.open(filename)
	image =  image.resize((int(screen_height*(0.75)), int(screen_height*(0.75))), Image.ANTIALIAS)
	img = ImageTk.PhotoImage(image)
	main_img = tk.Label(pw1,image=img, height=screen_height*(0.75), width =screen_width*(0.75))
	main_img.image = img 
	main_img.pack(side = "bottom", fill = "both", expand = "yes")

	# chanign iocn iamge
	org_picture.pack_forget()
	# image2 = Image.open(filename_copy)
	image =  image.resize((100, 100), Image.ANTIALIAS)
	img3 = ImageTk.PhotoImage(image)
	org_picture = tk.Label(pw3,image=img3, height=100, width =100)
	org_picture.image = img3 
	org_picture.pack(side = tk.LEFT, )

	#reseting everyting 
	negative_btn_label.pack_forget()
	negative_btn_fire_status = 0

	log_btn_label.pack_forget()
	log_btn_fire_status = 0

	power_btn_label.pack_forget()
	power_btn_fire_status = 0

	histogran_btn_label.pack_forget()
	histogran_btn_fire_status = 0
	print(filename)


m_top = PanedWindow(root,height=screen_height*(0.15))
m_top.pack(fill=BOTH,)

m_bottom = PanedWindow(root,height=screen_height*(0.75))
m_bottom.pack(fill=BOTH, )

m_status = PanedWindow(root,height=screen_height*(0.10))
m_status.pack(fill=BOTH, )
#configuring bottom 
pw1 = PanedWindow(m_bottom, height=screen_height*(0.75), width =screen_width*(0.75))
m_bottom.add(pw1) 
pw1.configure(background='white')

pw2=PanedWindow(m_bottom, height=screen_height*(0.25), width=screen_width*(0.25))
m_bottom.add(pw2) 
pw2.configure(background='red')

#adding to panel 1

image = Image.open(filename)
image =  image.resize((int(screen_height*(0.75)), int(screen_height*(0.75))), Image.ANTIALIAS)
img = ImageTk.PhotoImage(image)

main_img = tk.Label(pw1,image=img, height=screen_height*(0.75), width =screen_width*(0.75))
main_img.image = img 
main_img.pack(side = "bottom", fill = "both", expand = "yes")

p1 = PanedWindow()
pw2.add(p1)


#slection option 

selct_label = tk.Label(p1,text = "Select Method",font=("Helvetica", 20),bd=5)
selct_label.pack(fill=BOTH,)


negative_btn = tk.Button(p1, text="Negative",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
negative_btn.bind('<Button-1>', negative_btn_fire)
negative_btn.pack(fill=X, pady=0)

log_btn = tk.Button(p1, text="Log Transformation",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
log_btn.bind('<Button-1>', log_btn_fire)
log_btn.pack(fill=X, pady=0)

power_btn = tk.Button(p1, text="Power Law Transformation",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
power_btn.bind('<Button-1>', power_btn_fire)
power_btn.pack(fill=X, pady=0)

histogran_btn = tk.Button(p1, text="Histogran",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
histogran_btn.bind('<Button-1>', histogran_btn_fire)
histogran_btn.pack(fill=X, pady=0)



''' Here is new code  '''



transformDO = Transform.Transform()
input = Input.Input()

# def translate_btn_fire(event):
# 	global input
# 	input.method = "translate"
# 	update_input_data_for_trans()

# def affine_btn_fire(event):
# 	global input
# 	input.method = "affine"
# 	update_input_data_for_trans()

# def shear_btn_fire(event):
# 	global input
# 	input.method = "shear"
# 	update_input_data_for_trans()

# def perspective_btn_fire(event):
# 	global input
# 	input.method = "perspective"
# 	update_input_data_for_trans()

def polar_btn_fire(event):
	global input
	input.method = "polar"
	update_input_data_for_trans()

def logpolar_btn_fire(event):
	global input
	input.method = "logpolar"
	update_input_data_for_trans()

def update_input_data_for_trans():
	global image
	global main_img
	global filename
	global input
	global transformDO
	input.image = ut.load(filename)
    # input.method = "affine"
	input.interpolation = "bilinear"
	input.fx = 1
	input.fy = 0.5
	input.x = 100
	input.y = 100
	input.z = 300
	r = 90
	input.a11 = math.cos(math.radians(r))
	input.a12 = -math.sin(math.radians(r))
	input.a21 = math.sin(math.radians(r))
	input.a22 = math.cos(math.radians(r))
	input.r = math.radians(0)
	# if(input.image == None):
	# 	print("no image found") 
	# 	return
	result = transformDO.trans(input)
	print("iamge send to trans")
	#updating iamge
	main_img.pack_forget()
	cv_img = result					#storing to rbg filter
	cv_img =   cv2.resize(cv_img, (int(screen_height*(0.75)), int(screen_height*(0.75)) ))
	img = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
	main_img = tk.Label(pw1,image=img, height=screen_height*(0.75), width =screen_width*(0.75))
	main_img.image = img 
	main_img.pack(side = "bottom", fill = "both", expand = "yes")



# translate_btn = tk.Button(p1, text="translate")
# translate_btn.bind('<Button-1>', translate_btn_fire)
# translate_btn.pack(fill=X, pady=0)

# affine_btn = tk.Button(p1, text="affine Transformation",)
# affine_btn.bind('<Button-1>', affine_btn_fire)
# affine_btn.pack(fill=X, pady=0)

# shear_btn = tk.Button(p1, text="shear Law Transformation",)
# shear_btn.bind('<Button-1>', shear_btn_fire)
# shear_btn.pack(fill=X, pady=0)

# perspective_btn = tk.Button(p1, text="perspective")
# perspective_btn.bind('<Button-1>', perspective_btn_fire)
# perspective_btn.pack(fill=X, pady=0)

polar_btn = tk.Button(p1, text="polar Transformation",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
polar_btn.bind('<Button-1>', polar_btn_fire)
polar_btn.pack(fill=X, pady=0)

logpolar_btn = tk.Button(p1, text="logpolar",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
logpolar_btn.bind('<Button-1>', logpolar_btn_fire)
logpolar_btn.pack(fill=X, pady=0)




'''  end here new code  '''
#x y z 
labelframe1 = PanedWindow(p1)
labelframe1.pack( pady=10)
x_label = Label(labelframe1,text = "B:")
x_label.pack(side=LEFT)
T_R = Text(labelframe1, height=1, width=20,background="#005cb9",foreground="white",font=("Symbol", 15 ))
T_R.pack(side=LEFT)


labelframe2 = PanedWindow(p1)
labelframe2.pack( pady=10)
y_label = tk.Label(labelframe2,text = "R:")
y_label.pack(side=LEFT)
T_B = Text(labelframe2, height=1, width=20,background="#005cb9",foreground="white",font=("Symbol", 15 ))
T_B.pack(side=LEFT)


labelframe2 = PanedWindow(p1)
labelframe2.pack( pady=10)
z_label = tk.Label(labelframe2,text = "G:")
z_label.pack(side=LEFT)
T_G = Text(labelframe2, height=1, width=20,background="#005cb9",foreground="white",font=("Symbol", 15 ))
T_G.pack(side=LEFT)

labelframe2 = PanedWindow(p1)
labelframe2.pack( pady=10)
Button_apply_rbg = Button(labelframe2, text = "Apply", height=1, width=20,background="#005cb9",foreground="white",font=("Symbol", 15 ))
Button_apply_rbg.bind('<Button-1>',apply_rbg)
Button_apply_rbg.pack(side=LEFT)


#configuring top
pw3 = PanedWindow(m_top, height=screen_height*(0.25), width =screen_width*(0.75))
m_top.add(pw3) 
pw3.configure(background='blue')


# org_picture = tk.Label(pw3,text = "Orginal Img",font=("Helvetica", 30),justify=CENTER)
# org_picture.pack(side=tk.LEFT)
image = Image.open(filename)
image =  image.resize((100, 100), Image.ANTIALIAS)
img3 = ImageTk.PhotoImage(image)
org_picture = tk.Label(pw3,image=img3, height=100, width =100)
org_picture.image = img3 
org_picture.pack(side = tk.LEFT, )

browse_button = tk.Button(pw3, text="Browse Image",command=getfile,font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
browse_button.pack(side=tk.LEFT, padx=50)

#status show hide
pw5 = PanedWindow(m_status, height=screen_height*(0.10), width =screen_width)
m_status.add(pw5) 
pw5.configure(background='blue')



#intializeign text 
setTextRBG()
root.mainloop()