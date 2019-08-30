import os
import math
import argparse
import pickle

from PIL import Image
from multiprocessing import Process

import numpy as np
import cv2
import nori2 as nori
import scipy.io as sio

from tqdm import tqdm


def get_file_list(dir_path):
  print('Getting file list: ')
  file_list = []
  for dir_1 in os.listdir(dir_path):
    for dir_2 in os.listdir('{}/{}'.format(dir_path, dir_1)):
      for f in os.listdir(os.path.join(dir_path, dir_1, dir_2)):
        if f.endswith('.bin'):
          file_list.append(os.path.join(dir_path, dir_1, dir_2, f))

  print('Total: {}'.format(len(file_list)))
  return file_list


def get_l2_dist(point1, point2):
  '''
  :param point1: tuple (x, y) int or float
  :param point2: tuple (x, y)
  :return: float
  '''
  return float(((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5)


def qual_square(char_box):
  a = get_l2_dist(char_box[0], char_box[1])
  b = get_l2_dist(char_box[1], char_box[2])
  c = get_l2_dist(char_box[2], char_box[0])
  p = (a + b + c) / 2
  s = np.sqrt(p * (p - a) * (p - b) * (p - c))
  a = get_l2_dist(char_box[2], char_box[3])
  b = get_l2_dist(char_box[3], char_box[0])
  p = (a + b + c) / 2
  s += np.sqrt(p * (p - a) * (p - b) * (p - c))
  return s


def sample_filter(char_box, word, aspect_ratio):
  ## remove the sample if it has char_area 0 in it.
  char_box = np.array(char_box, dtype=np.int32)
  char_box = char_box.T.reshape((-1, 4, 2))
  char_box = np.clip(char_box, 0, math.inf)
  for i in range(len(char_box)):
    s = qual_square(char_box[i])
    if s == 0:
      return False
    v1 = get_l2_dist(char_box[0][0], char_box[0][1])
    v2 = get_l2_dist(char_box[0][1], char_box[0][2])
    v3 = get_l2_dist(char_box[0][2], char_box[0][3])
    v4 = get_l2_dist(char_box[0][3], char_box[0][0])
    if v1 <= 1 or v2 <= 1 or v3 <= 1 or v4 <= 1:
      return False
  flag = True
  if aspect_ratio < 0.2:
    return False
  if word == "":
    return False
  return True


def run_child(dest_path, file_list, job_no, margin_ratio, max_num):
  dest_path = '{}/{}.nori'.format(dest_path, job_no)
  if os.path.exists(dest_path):
    os.system('rm -r {0}'.format(dest_path))

  with nori.open(dest_path, 'w') as image_writer:
    count = 0
    for filename in file_list:
      count += 1
      if count > max_num:
        break
      if (count - 1) % 500 == 0:
        print('Job {}: total: {}, generated: {}'.format(job_no, max_num, count - 1))
      with open(filename, 'rb') as pklfile:
        pkl = pickle.load(pklfile, encoding='latin1')

      img = pkl['img'].copy()
      img_height, img_width, _ = img.shape

      word_bbs = np.array(pkl['contour'][1], dtype=np.int32)
      word_bbs = np.split(word_bbs, len(word_bbs), 0)
      word_bbs = [x.transpose([1, 0, 2]) for x in word_bbs]

      char_bbs = np.array(pkl['contour'][0], dtype=np.int32)
      chars = pkl['chars']

      char_bb_index = 0
      for i in range(len(word_bbs)):
        bb = word_bbs[i]
        word = chars[i]

        bb = np.squeeze(bb, axis=1)

        min_w, min_h = np.amin(bb, axis=0)
        max_w, max_h = np.amax(bb, axis=0)

        #margin = margin_ratio * np.sqrt((max_w - min_w) * (max_h - min_h))
        margin = 0
        min_w = int(round(max(min_w - margin * (np.random.rand() + 0.5), 0)))
        min_h = int(round(max(min_h - margin * (np.random.rand() + 0.5), 0)))
        max_w = int(round(min(max_w + margin * (np.random.rand() + 0.5), img_width - 1)))
        max_h = int(round(min(max_h + margin * (np.random.rand() + 0.5), img_height - 1)))

        char_bb = char_bbs[char_bb_index:char_bb_index + len(word)] #N, 4, 2
        char_bb_index += len(word)

        char_bb[:, :, ::2] = char_bb[:, :, ::2] - min_w
        char_bb[:, :, 1::2] = char_bb[:, :, 1::2] - min_h
        if not np.all(char_bb >= 0):
          continue

        img_cropped = img[min_h:max_h, min_w:max_w].copy()
        try:
          img_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
        except:
          continue
        char_box = char_bb.T
        words = ''.join(word)
        aspect_ratio = (max_w - min_w) / (max_h - min_h)
        
        if(sample_filter(char_box.tolist(), words, aspect_ratio)):
          image_writer.put(img_data, filename='', extra=dict(char_box=char_box.tolist(), words=words, aspect_ratio=aspect_ratio))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/unsullied/sharefs/_csg_algorithm/Interns/guanyushuo/OCR/SynthText/Synthtext/SynthTextData/results_bin',
                      help='Target tar file to impact into nori file')
  parser.add_argument('--dest_path', type=str,
                      default='/unsullied/sharefs/_csg_algorithm/Interns/yangmingkun/datasets/scenetext/noris/synthtext',
                      help='Destination file path to store the generated nori file')
  parser.add_argument('--jobs', type=int)
  parser.add_argument('--margin', type=float, default=0.1)
  parser.add_argument('--max_num', type=int)
  args = parser.parse_args()

  data_dir = args.data_dir
  dest_path = args.dest_path

  if os.path.exists(dest_path):
    os.system('rm -r {0}'.format(dest_path))

  file_list = get_file_list(data_dir)
  file_list_length = len(file_list)

  step = int(math.ceil(file_list_length / args.jobs))
  processes = []

  for i in range(args.jobs):
    begin = i * step
    end = min((i + 1) * step, file_list_length)
    if args.max_num is None:
      p = Process(target=run_child, args=(dest_path, file_list[begin:end], i, args.margin, end - begin))
    else:
      p = Process(target=run_child, args=(dest_path, file_list[begin:end], i, args.margin, args.max_num))
    p.daemon = True
    p.start()
    processes.append(p)

  for p in processes:
    p.join()


if __name__ == '__main__':
  main()
