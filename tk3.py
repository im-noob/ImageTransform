#using python 3
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
from image_transformer import ImageTransformer #for rotation 

from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image


''' updating code '''
from Implementation import Utility as ut
from Implementation import Input
from Implementation import Transform
import math
from Implementation import bothfunctions as bt

from tkinter import messagebox
from multiprocessing import Process

from scipy.ndimage.filters import median_filter
import time
import threading


transformDO = Transform.Transform()
input = Input.Input()



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
running = 0
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
		npcv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)		#correction cv2 problem 
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

def xyz_trans(event):
	global image
	global main_img
	global filename
	global input

	rotation_s = T_rotation.get('1.0', 'end-1c')
	
	if(len(rotation_s) == 0):
		angel = 0
	else:
		angel = int(rotation_s)

	
	
	img = cv2.imread(filename,0)

	rows,cols = img.shape

	M = cv2.getRotationMatrix2D((cols/2,rows/2),angel,1)
	dst = cv2.warpAffine(img,M,(cols,rows))

	#updating iamge
	main_img.pack_forget()
	cv_img = dst					#storing to rotating  filter
	cv_img =   cv2.resize(cv_img, (int(screen_height*(0.75)), int(screen_height*(0.75)) ))
	img = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
	main_img = tk.Label(pw1,image=img, height=screen_height*(0.75), width =screen_width*(0.75))
	main_img.image = img 
	main_img.pack(side = "bottom", fill = "both", expand = "yes")
	return


def apply_input(event):
	try:
		# global T_x
		global input
		X_s = T_X.get('1.0', 'end-1c')
		Y_s = T_Y.get('1.0', 'end-1c')
		Z_s = T_Z.get('1.0', 'end-1c')
		FX_s = T_fX.get('1.0', 'end-1c')
		FY_s = T_fY.get('1.0', 'end-1c')
		R_s = T_R.get('1.0', 'end-1c')
		
		a11_s = T_a11.get('1.0', 'end-1c')
		a12_s = T_a12.get('1.0', 'end-1c')
		a21_s = T_a21.get('1.0', 'end-1c')
		a22_s = T_a22.get('1.0', 'end-1c')


		if(len(X_s) == 0):
			X = 0
		else:
			X = int(X_s)

		if(len(Y_s) == 0):
			Y = 0
		else:
			Y = int(Y_s)

		if(len(Z_s) == 0):
			Z = 0
		else:
			Z = int(Z_s)

		if(len(FX_s) == 0):
			FX = 0
		else:
			FX = float(FX_s)
		
		if(len(FY_s) == 0):
			FY = 0
		else:
			FY = float(FY_s)
		
		if(len(R_s) == 0):
			R = 0
		else:
			R = int(R_s)

		#a11 a12 a21 a22
		if(len(a11_s) == 0):
			a11 = 0
		else:
			a11 = float(a11_s)

		if(len(a12_s) == 0):
			a12 = 0
		else:
			a12 = float(a12_s)

		if(len(a21_s) == 0):
			a21 = 0
		else:
			a21 = float(a21_s)

		if(len(a22_s) == 0):
			a22 = 0
		else:
			a22 = float(a22_s)

		input.fx = FX
		input.fy = FY
		input.r = R
		r = R
	    
		input.x = X
		input.y = Y
		input.z = Z

		input.a11 = a11
		input.a12 = a12
		input.a21 = a21
		input.a22 = a22
		
		print(X)
		print(Y)
		print(Z)
	except:
		running = 0
		messagebox.showinfo("warning", "Invalid Input")
		return

	# xyz_trans(X,Y,Z)
	return(R)

def setTextXYZ():
	T_X.insert('1.0', '100')
	T_Y.insert('1.0', '100')
	T_Z.insert('1.0', '300')
	T_fX.insert('1.0', '1')
	T_fY.insert('1.0', '0.5')
	T_R.insert('1.0', '90')
	T_a11.insert('1.0', '6.12')
	T_a12.insert('1.0', '-1.0')
	T_a21.insert('1.0', '1.0')
	T_a22.insert('1.0', '6.12')

	# T_scale.current(1)
    


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

selct_label = tk.Label(p1,text = "Select Method for transformation",font=("Fixedsys", 12,"bold"),bd=5)
selct_label.pack(fill=BOTH,)


negative_btn = tk.Button(p1, text="Negative",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
negative_btn.bind('<Button-1>', negative_btn_fire)
negative_btn.pack(fill=X, pady=0,padx=50)

log_btn = tk.Button(p1, text="Log Transformation",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
log_btn.bind('<Button-1>', log_btn_fire)
log_btn.pack(fill=X, pady=0,padx=50)

power_btn = tk.Button(p1, text="Power Law Transformation",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
power_btn.bind('<Button-1>', power_btn_fire)
power_btn.pack(fill=X, pady=0,padx=50)

# histogran_btn = tk.Button(p1, text="Histogran",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
# histogran_btn.bind('<Button-1>', histogran_btn_fire)
# histogran_btn.pack(fill=X, pady=0,padx=50)







''' Here is new code  '''




def translate_btn_fire(event):
	global inputv
	input.method = "translate"
	threading.Thread(target=call_update_input_data_for_trans).start()

def affine_btn_fire(event):
	global input
	input.method = "affine"
	threading.Thread(target=call_update_input_data_for_trans).start()

def shear_btn_fire(event):
	global input
	input.method = "shear"
	threading.Thread(target=call_update_input_data_for_trans).start()

def perspective_btn_fire(event):
	global input
	input.method = "perspective"
	threading.Thread(target=call_update_input_data_for_trans).start()

def polar_btn_fire(event):
	global inpupdate_input_data_for_transut
	input.method = "polar"

	threading.Thread(target=call_update_input_data_for_trans).start()

	# pr2=Process(target=update_input_data_for_trans, args=())
	# pr2.start()
	return

	

def logpolar_btn_fire(event):
	global input
	input.method = "logpolar"
	threading.Thread(target=call_update_input_data_for_trans).start()
	

def scale_btn_fire(event):
	global input
	print("scale button fire")
	input.method = "scale"
	# input.interpolation = "neighbor"
	opt = T_scale.get()
	if(opt == "neighbor"):
		input.interpolation = "neighbor"
	elif(opt == "lanczos4"):
		input.interpolation = "lanczos4"
	elif(opt == "bicubic"):
		input.interpolation = "bicubic"
	else:
		print("no method for scalling:"+opt)
		
	threading.Thread(target=call_update_input_data_for_trans).start()


#hisgorgram 	 
def streching_histogram(img):
	hist,bins = np.histogram(img.flatten(),256,[0,256])
	cdf = hist.cumsum()
	cdf_normalized = cdf * hist.max()/ cdf.max()
	plt.plot(cdf_normalized, color = 'b')
	plt.hist(img.flatten(),256,[0,256], color = 'r')
	plt.xlim([0,256])
	plt.legend(('cdf','histogram'), loc = 'upper left')
	messagebox.showinfo("warning", "im done...")
	plt.show()

def shaping_histogram(img):
	color = ('b','g','r')
	for i,col in enumerate(color):
	    histr = cv2.calcHist([img],[i],None,[256],[0,256])
	    plt.plot(histr,color = col)
	    plt.xlim([0,256])
	messagebox.showinfo("warning", "im done...")
	plt.show()



def histogram_btn_fire(event):
	global input
	print("histogram button fire")
	input.method = "histogram"
	# input.histogram = "neighbor"
	opt = T_histogram.get()
	if(opt == "Normal Histogram"):
		input.histogram = "normal_hist"
	elif(opt == "Histogram streching"):
		input.histogram = "streching_hist"
	elif(opt == "Histogram shaping"):
		input.histogram = "shaping_hist"
	else:
		print("no method for scalling:"+opt)
		
	threading.Thread(target=call_update_input_data_for_trans).start()

def doSharpness(event):
	original_image = plt.imread(filename).astype('uint16')
	# Convert to grayscale
	gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
	# Median filtering
	gray_image_mf = median_filter(gray_image, 1)
	# Calculate the Laplacian
	lap = cv2.Laplacian(gray_image_mf,cv2.CV_64F)
	# Calculate the sharpened image
	sharp = gray_image - 0.9*lap
	# messagebox.showinfo("warning", "im done...")
	cv2.imshow('image',sharp)
	while True:
		k = cv2.waitKey(100) 
		# change the value from the original 0 (wait forever) to something appropriate
		if k == 27:
			print('ESC')
			cv2.destroyAllWindows()
			break        
		if cv2.getWindowProperty('image',cv2.WND_PROP_VISIBLE) < 1:        
			break 
	cv2.destroyAllWindows()
	return

def unsharp_btn_fire(event):
	global input
	input.method = "unsharp"
	threading.Thread(target=call_update_input_data_for_trans).start()


def call_update_input_data_for_trans():
	global msgBOx
	global running
	if(running == 1):
		messagebox.showinfo("warning", "It's still calculating...")
		return
	else:
		running = 1
		threading.Thread(target=update_input_data_for_trans).start()
		messagebox.showinfo("warning", "wait until im done...")


def update_input_data_for_trans():
	global image
	global main_img
	global filename
	global input
	global transformDO
	global msgBOx
	global running
	try:
		# updaing input 
		r = apply_input(11)
		input.r_value = r
		input.image = ut.load(filename)
		# input.method = "affine"
		print("method :"+input.method)
		# print("img :"+input.image)
		print(input.fx)
		print(input.fy)
		print(input.r)
		print("interpolation")
		print(input.interpolation)
	    
	    #ensuring all value done
		"""initilize everything before calling"""
		# print('method: ' + input.method) print('image: ' + input.image)
		# print('interpolation: ' + input.interpolation) print('fx: ' + input.fx)
		# print('fy: ' + input.fy) print('x: ' + input.x) print('y: ' + input.y)
		# print('z: ' + input.z) print('r: ' + input.r)



		# input.a11 = math.cos(math.radians(r))
		# input.a12 = -math.sin(math.radians(r))
		# input.a21 = math.sin(math.radians(r))
		# input.a22 = math.cos(math.radians(r))
		input.r = math.radians(0)
		# if(input.image == None):
		# 	print("no image found") 
		# 	return
		if(input.method == "scale" and input.interpolation =="bicubic"):
			result = bt.bicubic_interpolation(input.image,input.fx,input.fy,.5)
		elif(input.method == "scale" and input.interpolation =="lanczos4"):
			result = bt.lanczos4_interpolation(input.image,input.fx,input.fy)
		# elif(input.method == "unsharp"):
		# 	doSharpness(0.9)
		elif(input.method == "histogram" and input.histogram =="normal_hist"):
			histogran_btn_fire("event")
		elif(input.method == "histogram" and input.histogram =="streching_hist"):
			streching_histogram(input.image)
		elif(input.method == "histogram" and input.histogram =="shaping_hist"):
			shaping_histogram(input.image)
		else:
			result = transformDO.trans(input)


		#for histogram
		
		if(input.method == "histogram" or input.method == "unsharp"):
			running = 0 
			return

		print("iamge send to trans")
		
		messagebox.showinfo("warning", "im done...")

		if(input.method == "scale" ):
			running = 0
			plt.imshow(result)
			plt.show()
			return
		#updating iamge
		main_img.pack_forget()
		cv_img = result					#storing to rbg filter
		cv_img =   cv2.resize(cv_img, (int(screen_height*(0.75)), int(screen_height*(0.75)) ))
		img = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
		main_img = tk.Label(pw1,image=img, height=screen_height*(0.75), width =screen_width*(0.75))
		main_img.image = img 
		main_img.pack(side = "bottom", fill = "both", expand = "yes")
		running = 0
	except:
		running = 0 
		messagebox.showinfo("try again")



translate_btn = tk.Button(p1, text="translate",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
translate_btn.bind('<Button-1>', translate_btn_fire)
translate_btn.pack(fill=X, pady=0,padx=50)

affine_btn = tk.Button(p1, text="affine Transformation",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
affine_btn.bind('<Button-1>', affine_btn_fire)
affine_btn.pack(fill=X, pady=0,padx=50)

shear_btn = tk.Button(p1, text="shear Law Transformation",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
shear_btn.bind('<Button-1>', shear_btn_fire)
shear_btn.pack(fill=X, pady=0,padx=50)

perspective_btn = tk.Button(p1, text="perspective",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
perspective_btn.bind('<Button-1>', perspective_btn_fire)
perspective_btn.pack(fill=X, pady=0,padx=50)



polar_btn = tk.Button(p1, text="polar Transformation",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
polar_btn.bind('<Button-1>', polar_btn_fire)
polar_btn.pack(fill=X, pady=0,padx=50)

logpolar_btn = tk.Button(p1, text="logpolar",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
logpolar_btn.bind('<Button-1>', logpolar_btn_fire)
logpolar_btn.pack(fill=X, pady=0,padx=50)

rotate_btn = tk.Button(p1, text="rotate",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
rotate_btn.bind('<Button-1>', xyz_trans)
rotate_btn.pack(fill=X, pady=0,padx=50)

unsharp_btn = tk.Button(p1, text="unsharp",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
unsharp_btn.bind('<Button-1>', doSharpness)
unsharp_btn.pack(fill=X, pady=0,padx=50)

# scale_btn = tk.Button(p1, text="scale",font=("Symbol", 12 ),background="#a8d",activebackground="#ffa500")
# scale_btn.bind('<Button-1>', scale_btn_fire)
# scale_btn.pack(fill=X, pady=0)

labelscale = PanedWindow(p1)
labelscale.pack()
scale_label = Label(labelscale,text = "Scale:")
scale_label.pack(side=LEFT)
T_scale = ttk.Combobox(labelscale, 
                            values=[
                                    "neighbor", 
                                    "lanczos4",
                                    "bicubic"], width=20)
T_scale.bind("<<ComboboxSelected>>", scale_btn_fire)
T_scale.pack(side=LEFT)





#new code again fo histogram

labelhistogram = PanedWindow(p1)
labelhistogram.pack()
histogram_label = Label(labelhistogram,text = "histogram:")
histogram_label.pack(side=LEFT)
T_histogram = ttk.Combobox(labelhistogram, 
                            values=[
                                    "Normal Histogram", 
                                    "Histogram streching",
                                    "Histogram shaping"], width=20)
T_histogram.bind("<<ComboboxSelected>>", histogram_btn_fire)
T_histogram.pack(side=LEFT)


'''  end here new code  '''






#x y z 
labelframe1 = PanedWindow(p1)
labelframe1.pack()
x_label = Label(labelframe1,text = "X:",font=("Symbol", 12 ))
x_label.pack(side=LEFT)
T_X = Text(labelframe1, height=1, width=20,background="#005cb9",foreground="white",font=("Symbol", 15 ))
T_X.pack(side=LEFT)

labelframe2 = PanedWindow(p1)
labelframe2.pack()
y_label = tk.Label(labelframe2,text = "Y:",font=("Symbol", 12 ))
y_label.pack(side=LEFT)
T_Y = Text(labelframe2, height=1, width=20,background="#005cb9",foreground="white",font=("Symbol", 15 ))
T_Y.pack(side=LEFT)


labelframe3 = PanedWindow(p1)
labelframe3.pack()
z_label = tk.Label(labelframe3,text = "Z:",font=("Symbol", 12 ))
z_label.pack(side=LEFT)
T_Z = Text(labelframe3, height=1, width=20,background="#005cb9",foreground="white",font=("Symbol", 15 ))
T_Z.pack(side=LEFT)

#fx fy r 
labelframe4 = PanedWindow(p1)
labelframe4.pack()
fx_label = Label(labelframe4,text = "FX:",font=("Symbol", 12 ))
fx_label.pack(side=LEFT)
T_fX = Text(labelframe4, height=1, width=20,background="#005cb9",foreground="white",font=("Symbol", 15 ))
T_fX.pack(side=LEFT)

labelframe5 = PanedWindow(p1)
labelframe5.pack()
fy_label = tk.Label(labelframe5,text = "FY:",font=("Symbol", 12 ))
fy_label.pack(side=LEFT)
T_fY = Text(labelframe5, height=1, width=20,background="#005cb9",foreground="white",font=("Symbol", 15 ))
T_fY.pack(side=LEFT)


labelframe6 = PanedWindow(p1)
labelframe6.pack()
z_label = tk.Label(labelframe6,text = "R:",font=("Symbol", 12 ))
z_label.pack(side=LEFT)
T_R = Text(labelframe6, height=1, width=20,background="#005cb9",foreground="white",font=("Symbol", 15 ))
T_R.pack(side=LEFT)


labelframe7 = PanedWindow(p1)
labelframe7.pack()
rotation_label = tk.Label(labelframe7,text = "angel:",font=("Symbol", 12 ))
rotation_label.pack(side=LEFT)
T_rotation = Text(labelframe7, height=1, width=20,background="#005cb9",foreground="white",font=("Symbol", 15 ))
T_rotation.pack(side=LEFT)

# labelframe7 = PanedWindow(p1)
# labelframe7.pack( pady=10)
# Button_apply_rbg = Button(labelframe7, text = "Rotate", height=1, width=20)
# Button_apply_rbg.bind('<Button-1>',apply_input)
# Button_apply_rbg.pack(side=LEFT)


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



#a11 a12 a21 a22 
labelframe8 = PanedWindow(pw3)
labelframe8.pack()
a11_label = Label(labelframe8,text = "a11:",font=("Symbol", 12 ))
a11_label.pack(side=LEFT)
T_a11 = Text(labelframe8, height=1, width=20,background="#005cb9",foreground="white",font=("Symbol", 15 ))
T_a11.pack(side=LEFT)

labelframe9 = PanedWindow(pw3)
labelframe9.pack()
a12_label = tk.Label(labelframe9,text = "a12:",font=("Symbol", 12 ))
a12_label.pack(side=LEFT)
T_a12 = Text(labelframe9, height=1, width=20,background="#005cb9",foreground="white",font=("Symbol", 15 ))
T_a12.pack(side=LEFT)

labelframe10 = PanedWindow(pw3)
labelframe10.pack()
a21_label = tk.Label(labelframe10,text = "a21:",font=("Symbol", 12 ))
a21_label.pack(side=LEFT)
T_a21 = Text(labelframe10, height=1, width=20,background="#005cb9",foreground="white",font=("Symbol", 15 ))
T_a21.pack(side=LEFT)

labelframe11 = PanedWindow(pw3)
labelframe11.pack()
a22_label = tk.Label(labelframe11,text = "a22:",font=("Symbol", 12 ))
a22_label.pack(side=LEFT)
T_a22 = Text(labelframe11, height=1, width=20,background="#005cb9",foreground="white",font=("Symbol", 15 ))
T_a22.pack(side=LEFT)





#status show hide
pw5 = PanedWindow(m_status, height=screen_height*(0.10), width =screen_width)
m_status.add(pw5) 
pw5.configure(background='blue')




#intializeign text 
setTextXYZ()
root.mainloop()