# Split dataset between train and val folders

import glob
import random
import os
import sys
from pathlib import Path
import shutil
import argparse

# Define and parse user input arguments

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', help='Path to data folder containing image and annotation files',
                    required=True)
parser.add_argument('--train_pct', help='Ratio of images to go to train folder; \
                    the rest go to validation folder (example: ".8")',
                    default=.8)

args = parser.parse_args()

data_path = args.datapath
train_percent = float(args.train_pct)

# Check for valid entries
if not os.path.isdir(data_path):
   print('Directory specified by --datapath not found. Verify the path is correct (and uses double back slashes if on Windows) and try again.')
   sys.exit(0)
if train_percent < .01 or train_percent > 0.99:
   print('Invalid entry for train_pct. Please enter a number between .01 and .99.')
   sys.exit(0)
val_percent = 1 - train_percent

# Define paths to image and annotation folders
cwd = os.getcwd()
train_img_path = os.path.join(cwd,'data/train/images')
train_txt_path = os.path.join(cwd,'data/train/labels')
val_img_path = os.path.join(cwd,'data/validation/images')
val_txt_path = os.path.join(cwd,'data/validation/labels')

for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
   if not os.path.exists(dir_path):
      os.makedirs(dir_path)
      print(f'Created folder at {dir_path}.')

# Get list of all images and annotation files
file_list = [path for path in Path(data_path).rglob('*')]

# Move .txt files and image files to seperate lists
txt_file_list = []
img_file_list = []
for fn in file_list:
  if fn.suffix == '.txt':
    txt_file_list.append(fn)
  elif fn.suffix in ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']:
    img_file_list.append(fn)

print(f'Number of image files: {len(img_file_list)}')
print(f'Number of annotation files: {len(txt_file_list)}')

# Determine number of files to move to each folder
file_num = len(img_file_list)
train_num = int(file_num*train_percent)
val_num = file_num - train_num
print('Images being copied to train: %d' % train_num)
print('Images being copied to validation: %d' % val_num)

# Select files randomly and copy them to train folder
for i in range(train_num):
    copy_me = random.choice(img_file_list)
    fn = copy_me.name
    base_fn = copy_me.stem
    parent_path = copy_me.parent
    txt_fn = base_fn + '.txt'
    shutil.copy(copy_me, train_img_path+'/'+fn)
    shutil.copy(os.path.join(parent_path,txt_fn),os.path.join(train_txt_path,txt_fn))
    img_file_list.remove(copy_me)

# Copy remaining files to validation folder
for i in range(val_num):
    copy_me = random.choice(img_file_list)
    fn = copy_me.name
    base_fn = copy_me.stem
    parent_path = copy_me.parent
    txt_fn = base_fn + '.txt'
    shutil.copy(copy_me, val_img_path+'/'+fn)
    shutil.copy(os.path.join(parent_path,txt_fn),os.path.join(val_txt_path,txt_fn))
    img_file_list.remove(copy_me)
