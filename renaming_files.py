"""A short scrip to rename the files in the database in incremental order, so they have their own unique and simple name"""

import glob
import os


directory = os.listdir("D:\\University\\Dissertation\\Database\\dave\\")
counter = 1

for folders in directory:
    folder = os.listdir("D:\\University\\Dissertation\\Database\\dave\\" + folders)
    os.chdir("D:\\University\\Dissertation\\Database\\dave\\" + folders)
    for filename in folder:
        os.rename(filename, str(counter) + ".JPG")
        counter += 1

