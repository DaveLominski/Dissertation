from shutil import *
import os
import glob
import re

source = os.listdir("D:\\University\\Dissertation\\Database\\KDEF")

for path in glob.glob("D:\\University\\Dissertation\\Database\\KDEF\\*\\*"):
    #dirname, filename = os.path.split(path)
    #print(path)
    #print(dirname, filename)
    #print(os.listdir(dirname))
    #a = os.listdir(dirname)
    #print(a)


    if "AFS" in path:
        print("LOL")
        move(path, ("D:\\University\\Dissertation\\Database\\sorted_images\\afraid"))

    if "AN" in path:
        move(path, ("D:\\University\\Dissertation\\Database\\sorted_images\\angry"))

    if "DI" in path:
        move(path, ("D:\\University\\Dissertation\\Database\\sorted_images\\disgusted"))

    if "HA" in path:
        move(path, ("D:\\University\\Dissertation\\Database\\sorted_images\\happy"))

    if "NE" in path:
        move(path, ("D:\\University\\Dissertation\\Database\\sorted_images\\neutral"))

    if "SA" in path:
        move(path, ("D:\\University\\Dissertation\\Database\\sorted_images\\sad"))

    if "SU" in path:
        move(path, ("D:\\University\\Dissertation\\Database\\sorted_images\\surprised"))

    

    
