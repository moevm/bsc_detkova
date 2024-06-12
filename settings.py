import argparse
import os
from datetime import datetime
from tqdm import tqdm
from tqdm.contrib import itertools
import json
import random as rnd
import matplotlib.pyplot as plt
import open3d as o3d
import calendar
import time
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score, jaccard_score, \
    ConfusionMatrixDisplay
from prettytable import PrettyTable

# -----------------------------------------------------------------------------
# S3DIS: ИНФОРМАЦИЯ О ДАТАСЕТЕ
# -----------------------------------------------------------------------------

movable_objects_set = {"board", "bookcase", "chair", "table", "sofa"}
structural_objects_set = {"ceiling", "door", "floor", "wall", "beam", "column", "window", "stairs"}
objects_set = structural_objects_set.union(movable_objects_set)

# Распределение того, как области делятся на train, val, test
training_areas = ["Area_1", "Area_2", "Area_3", "Area_4"]
val_areas = ["Area_5"]
test_areas = ["Area_6"]
all_areas = training_areas + val_areas + test_areas

# Объект, который будем визуализировать после сегментации
# Вручную выбран "table" - стол
segmentation_target_object = "table"

# -----------------------------------------------------------------------------
# СРЕДА И ПАРАМЕТРЫ МОДЕЛИ
# -----------------------------------------------------------------------------
# Параметры среды (файлы системы и прочие)
eparams = {
    'pc_data_path': "models",
    'pc_file_extension': ".txt",
    'pc_file_extension_rgb_norm': "_rgb_norm.txt",
    'pc_file_extension_sem_seg_suffix': "_annotated",
    's3dis_summary_file': "s3dis_summary.csv",
    'checkpoints_folder': "checkpoints",
    'checkpoint_name': "checkpoint_",
    'spaces_file': "spaces.json",
    'objects_file': "objects.json",
    'sliding_windows_folder': "sliding_windows",
    'camera_point_views': "camera"
}

# Model hyperparameters - гиперпараметры модели
hparams = {
    'epochs': 5,
    'batch_size': 32,
    'learning_rate': 0.001,
    'regularization_weight_decay_lambda': 1e-5,
    'num_classes': len(objects_set),
    'num_workers': None,
    # Необходимо для сжатия облака точек до единого размера
    'num_points_per_room': 4096,
    # Размерность, которая используется из датасета (3 (xyz) или 6 (xyzrgb) + цвета)
    'dimensions_per_object': 6,
    # Параметры для скользящих окон
    'win_width': 1,
    'win_depth': 1,
    'win_height': 4,
    'overlap': 0.5,  # 0-1, 0.5 - половина перекрытия окна. 1 создаст бесконечный цикл
    'window_filling': 0.9  # 0-1 часть скользящего окна, которую необходимо заполнить, чтобы оно не было выброшено
}

hparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
# Num workers больше 0 не работает с графическими процессорами
# (требуется распараллеливание кодирования)
max_workers = 2
hparams['num_workers'] = max_workers if hparams['device'] == 'cpu' else 0

# -----------------------------------------------------------------------------
# СОЗДАНИЕ ФАЙЛОВ И ПАПОК ДЛЯ ХРАНЕНИЯ ДАННЫХ
# -----------------------------------------------------------------------------
# Для хранения чекпоинтов
checkpoint_folder = os.path.join(eparams["pc_data_path"], eparams["checkpoints_folder"])
if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)

checkpoints = list(folder for folder in sorted(os.listdir(checkpoint_folder)))
if (len(checkpoints) == 0) :
    last_checkpoint = "-1"
else:
    last_checkpoint = checkpoints[-1]
last_ch_numb = int(last_checkpoint.replace(eparams["checkpoint_name"], "").replace(".pth", ""))
eparams["checkpoint_name"] += str(last_ch_numb + 1) + ".pth"

# Для хранения скользящих окон
path_to_root_sliding_windows_folder = os.path.join(eparams["pc_data_path"],
                                                   eparams['sliding_windows_folder'])
if not os.path.exists(path_to_root_sliding_windows_folder):
    os.makedirs(path_to_root_sliding_windows_folder)

# Папка будет создаваться в соответствии с этим соглашением: w_X_d_Y_h_Z_o_T
chosen_params = 'w' + str(hparams['win_width'])
chosen_params += '_d' + str(hparams['win_depth'])
chosen_params += '_h' + str(hparams['win_height'])
chosen_params += '_o' + str(hparams['overlap'])

path_to_current_sliding_windows_folder = os.path.join(
    path_to_root_sliding_windows_folder,
    chosen_params)

if not os.path.exists(path_to_current_sliding_windows_folder):
    os.makedirs(path_to_current_sliding_windows_folder)

camera_folder = os.path.join(eparams["pc_data_path"], eparams["camera_point_views"])

# -----------------------------------------------------------------------------
# ВИЗУАЛИЗАЦИЯ
# -----------------------------------------------------------------------------
# Целевая комната для визуализации, если не выбран случайный режим
# Комнаты с большим количеством книжных шкафов и досок
target_room_for_visualization = "Area_6_office_10"

# Get parameters from camera point of views for room of study
parameters_camera = o3d.io.read_pinhole_camera_parameters(os.path.join(camera_folder, "camera_settings.json"))

cparams = {
    'Red': [1,0,0],
    'Lime': [0,1,0],
    'Blue': [0,0,1],
    'Yellow': [1,1,0],
    'Cyan': [0,1,1],
    'Magenta': [1,0,1],
    'Dark_green': [0,0.39,0],
    'Deep_sky_blue': [0,0.75,1],
    'Saddle_brown': [0.54,0.27,0.07],
    'Lemon_chiffon': [1,0.98,0.8],
    'Turquoise': [0.25,0.88,0.81],
    'Gold': [1,0.84,0],
    'Orange': [1,0.65,0],
    'Chocolate': [0.82,0.41,0.12],
    'Peru': [0.8,0.52,0.25],
    'Blue_violet': [0.54,0.17,0.88],
    'Dark_grey': [0.66,0.66,0.66],
    'Grey': [0.5,0.5,0.5],
}

vparams = {
    'str_object_to_visualize': "chair",
    'num_max_points_from_GT_file': 50000,
    'num_max_points_1_object_model': 50000,
    'board_color': cparams['Red'],
    'bookcase_color': cparams['Lime'],
    'chair_color': cparams['Blue'],
    'table_color': cparams['Yellow'],
    'sofa_color': cparams['Cyan'],
    'ceiling_color': cparams['Magenta'],
    'clutter_color': cparams['Dark_green'],
    'door_color': cparams['Deep_sky_blue'],
    'floor_color': cparams['Saddle_brown'],
    'wall_color': cparams['Lemon_chiffon'],
    'beam_color': cparams['Turquoise'],
    'column_color': cparams['Gold'],
    'window_color': cparams['Orange'],
    'stairs_color': cparams['Chocolate'],
}

# ------------------------------------------------------------------------------
# СОЗДАНИЕ ПАРСЕРА ПРОГРАММЫ
# ------------------------------------------------------------------------------
# Парсер
parser_desc = ("Предоставляет удобные готовые опции для обучения и тестирования "
               "модели PointNet, которая выполняет сегментацию 3D облаков точек на наборе данных S3DIS ")

parser = argparse.ArgumentParser(prog="main",
                                 usage="%(prog)s.py task=(train|test|validation|watch) " +
                                       "objects=(movable|structural|all)",
                                 description=parser_desc)

parser.add_argument("--task",
                    "-t",
                    metavar="task",
                    type=str,
                    action="store",
                    nargs=1,
                    default="train",
                    choices=["train", "validation", "test", "watch"],
                    help="Задача: train, validate, test или watch")

parser.add_argument("--objects",
                    "-o",
                    metavar="objects",
                    type=str,
                    action="store",
                    nargs=1,
                    default="all",
                    choices=["movable", "structural", "all"],
                    help="Целевые объекты сегментации: movable, structural или all")

# Get parser args to decide what the program has to do
args = parser.parse_args()

# Adapt params depending on the target objects we're going to work
if "movable" in args.objects:
    objects_set = movable_objects_set

if "structural" in args.objects:
    objects_set = structural_objects_set

if "all" in args.objects:
    objects_set = structural_objects_set.union(movable_objects_set)

hparams["num_classes"] = len(objects_set)
