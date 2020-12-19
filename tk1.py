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

# m_status = PanedWindow(root,height=screen_height*(0.10))
# m_status.pack(fill=BOTH, )
#configuring bottom 
pw1 = PanedWindow(m_bottom, height=screen_height*(0.75), width =screen_width*(0.75))
m_bottom.add(pw1) 
pw1.configure(background='white')

# pw2=PanedWindow(m_bottom, height=screen_height*(0.25), width=screen_width*(0.25))
# m_bottom.add(pw2) 
# pw2.configure(background='red')

#adding to panel 1

image = Image.open(filename)
image =  image.resize((int(screen_height*(0.75)), int(screen_height*(0.75))), Image.ANTIALIAS)
img = ImageTk.PhotoImage(image)

main_img = tk.Label(pw1,image=img, height=screen_height*(0.75), width =screen_width*(0.75))
main_img.image = img 
main_img.pack(side = "bottom", fill = "both", expand = "yes")

# p1 = PanedWindow()
# pw2.add(p1)


#slection option 

# selct_label = tk.Label(p1,text = "Select Method",font=("Helvetica", 20),bd=5)
# selct_label.pack(fill=BOTH,)


# negative_btn = tk.Button(p1, text="Negative")
# negative_btn.bind('<Button-1>', negative_btn_fire)
# negative_btn.pack(fill=X, pady=10)

# log_btn = tk.Button(p1, text="Log Transformation",)
# log_btn.bind('<Button-1>', log_btn_fire)
# log_btn.pack(fill=X, pady=10)

# power_btn = tk.Button(p1, text="Power Law Transformation",)
# power_btn.bind('<Button-1>', power_btn_fire)
# power_btn.pack(fill=X, pady=10)

# histogran_btn = tk.Button(p1, text="Histogran",)
# histogran_btn.bind('<Button-1>', histogran_btn_fire)
# histogran_btn.pack(fill=X, pady=10)

# #x y z 
# labelframe1 = PanedWindow(p1)
# labelframe1.pack( pady=10)
# x_label = Label(labelframe1,text = "B:")
# x_label.pack(side=LEFT)
# T_R = Text(labelframe1, height=1, width=20)
# T_R.pack(side=LEFT)


# labelframe2 = PanedWindow(p1)
# labelframe2.pack( pady=10)
# y_label = tk.Label(labelframe2,text = "R:")
# y_label.pack(side=LEFT)
# T_B = Text(labelframe2, height=1, width=20)
# T_B.pack(side=LEFT)


# labelframe2 = PanedWindow(p1)
# labelframe2.pack( pady=10)
# z_label = tk.Label(labelframe2,text = "G:")
# z_label.pack(side=LEFT)
# T_G = Text(labelframe2, height=1, width=20)
# T_G.pack(side=LEFT)

# labelframe2 = PanedWindow(p1)
# labelframe2.pack( pady=10)
# Button_apply_rbg = Button(labelframe2, text = "Apply", height=1, width=20)
# Button_apply_rbg.bind('<Button-1>',apply_rbg)
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

browse_button = tk.Button(pw3, text="Browse Image",command=getfile)
browse_button.pack(side=tk.LEFT, padx=50)

# #status show hide
# pw5 = PanedWindow(m_status, height=screen_height*(0.10), width =screen_width)
# m_status.add(pw5) 
# pw5.configure(background='blue')



# #intializeign text 
# setTextRBG()
root.mainloop()