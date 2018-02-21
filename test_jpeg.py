#!/usr/bin/env python

import os
from PIL import Image
import argparse
from subprocess import call


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='coco-animals/train')
parser.add_argument('--val_dir', default='coco-animals/val')


VGG_MEAN = [123.68, 116.78, 103.94]


def list_images(directory):
  """
  Get all the images and labels in directory/label/*.jpg
  """
  labels = os.listdir(directory)
  # Sort the labels so that training and validation get them in the same order
  labels.sort()

  files_and_labels = []
  for label in labels:
    for f in os.listdir(os.path.join(directory, label)):
      files_and_labels.append((os.path.join(directory, label, f), label))
      try:
        I = Image.open(os.path.join(directory, label, f))
      except:
        f1 = os.path.join(directory, label, f)
        #f2= os.path.join(directory, label, f[1:])

        print("unrecognized image: {}".format(f1))
        #print("new image {}".format(f2))
        call(["rm","-f",f1])

  filenames, labels = zip(*files_and_labels)
  filenames = list(filenames)
  labels = list(labels)
  unique_labels = list(set(labels))
  print(unique_labels)

  label_to_int = {}
  for i, label in enumerate(unique_labels):
    label_to_int[label] = i

  labels = [label_to_int[l] for l in labels]

  return filenames, labels


def main(args):
  # Get the list of filenames and corresponding list of labels for training et validation
  train_filenames, train_labels = list_images(args.train_dir)
  val_filenames, val_labels = list_images(args.val_dir)
  results = "res.txt"
  with open(results,"a") as r:
    for i in train_filenames:
      r.write("{}\n".format(i))

  # print(train_labels)
  # print(val_labels)
  assert set(train_labels) == set(val_labels), \
    "Train and val labels don't correspond:\n{}\n{}".format(set(train_labels),
                                                            set(val_labels))

  num_classes = len(set(train_labels))
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
