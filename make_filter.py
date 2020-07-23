import glob
import numpy as np

from project import MACE_filter

user = input('select the user name: ')

path = ('image_data/'+ user +'/*.jpg')

A = MACE_filter(path)

np.savetxt('/filtros/' + user + '.txt',A)
