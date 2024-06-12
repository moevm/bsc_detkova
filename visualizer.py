from settings import *
from dataset import *


def render_segmentation(dict_to_use={},
                        str_area_and_office="",
                        dict_model_segmented_points={},
                        b_multiple_seg=False,
                        b_hide_wall=True,
                        draw_original_rgb_data=False,
                        b_show_room_points=True):
    """
    Функция для визуализации сегментации объекта, сгенерированной как на
    основе модели, так и на основе реальных данных.

    Функция после визуализации создает файл png для каждого сгенерированного изображения.
    Аргументы:
        dict_to_use: Словарь с объектами для обнаружения
        str_area_and_office: String, область и офис
        dict_model_segmented_points: Dict, Словарь с прогнозируемыми точками каждого объекта
            для каждого скользящего окна.
        b_multiple_seg: bool, чтобы визуализировать все объекты, указанные в параметре dict_to_use
        b_hide_wall: bool, скрывает точки, соответствующие стене
        draw_original_rgb_data: bool, Для отображения исходного rgb-цвета тензора данных.
        b_show_room_points: bool, точки демонстрационного зала,
            выделены серым цветом, если для параметра draw_original_rgb_data задано значение False
    """
    torch.set_printoptions(profile="full")

    # Открыть аннотированный файл комнаты с GT сегментированными точками
    a = str_area_and_office.split('_')
    str_area = a[0] + "_" + a[1]
    str_room = a[2] + "_" + a[3]

    path_to_space = os.path.join(eparams["pc_data_path"], str_area,
                                 str_room)  # \Area_1\conferenceRoom_1
    sem_seg_file = str_room + eparams["pc_file_extension_sem_seg_suffix"] + eparams[
        "pc_file_extension"]  # conferenceRoom_1_annotated.txt
    path_to_room_annotated_file = os.path.join(path_to_space,
                                               sem_seg_file)  # \Area_1\conferenceRoom_1\conferenceRoom_1_annotated.txt

    data_gt = np.genfromtxt(path_to_room_annotated_file,
                            dtype=float,
                            skip_header=1,
                            delimiter='',
                            names=None)

    # Установить максимальное количество точек для каждого файла GT
    room_points_gt = PointSampler(data_gt, vparams["num_max_points_from_GT_file"]).sample()

    # создать словарь объектов со всеми точками из всех скользящих окон
    dict_of_tensors_allpoints_per_object = {}
    for slid_wind_key, slid_wind_value in dict_model_segmented_points.items():
        for object in slid_wind_value:
            if object[-1].numel() != 0:  # Проверка, что тензор не пуст
                # Визуализация для всех объектов
                if b_multiple_seg:
                    if object[0] in dict_of_tensors_allpoints_per_object.keys():  # Проверка, что объект уже в dict
                        dict_of_tensors_allpoints_per_object[object[0]] = \
                            torch.cat((dict_of_tensors_allpoints_per_object[object[0]], object[-1]), 0)
                    # если не в словаре - сохранить
                    else:
                        dict_of_tensors_allpoints_per_object[object[0]] = object[-1]

                # Визуализация только одного объекта
                elif object[0] == vparams["str_object_to_visualize"]:
                    if object[0] in dict_of_tensors_allpoints_per_object.keys():  # Проверка, что объект уже в dict
                        dict_of_tensors_allpoints_per_object[object[0]] = \
                            torch.cat((dict_of_tensors_allpoints_per_object[object[0]], object[-1]), 0)
                    # если не в словаре - сохранить
                    else:
                        dict_of_tensors_allpoints_per_object[object[0]] = object[-1]

    if len(dict_of_tensors_allpoints_per_object.keys()) == 0:
        print(80 * "-")
        print("Модель не определила точек для объекта " + vparams["str_object_to_visualize"])
        print(80 * "-")
        return

    # уменьшить количество точек на объект
    dict_of_tensors_allpoints_per_object_reduced = {}
    for k_object, v_object in dict_of_tensors_allpoints_per_object.items():
        dict_of_tensors_allpoints_per_object_reduced[k_object] = \
            PointSampler(v_object, vparams["num_max_points_1_object_model"]).sample()

    # -----------------------------------------------------------
    # GROUND TRUTH - ИСТИНА (создание облаков точек для gt файла)
    # -----------------------------------------------------------
    all_pointcloud_object_gt = []
    for k_object_name, v_object_index in dict_to_use.items():
        if b_multiple_seg:
            if b_hide_wall and k_object_name == "wall":
                continue
            # Получить координаты точки, совпадающие с названием объекта
            points = room_points_gt[(room_points_gt[:, 6] == v_object_index).nonzero().squeeze(1)]
            # Создать облако точек, чтобы отрисовать его позже
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points[:, :3])  # получить координаты xyz
            pc.paint_uniform_color(vparams[k_object_name + '_color'])
            all_pointcloud_object_gt.append(pc)
        elif k_object_name == vparams["str_object_to_visualize"]:
            points = room_points_gt[(room_points_gt[:, 6] == v_object_index).nonzero().squeeze(1)]
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points[:, :3])  # получить координаты xyz
            pc.paint_uniform_color(vparams[k_object_name + '_color'])
            all_pointcloud_object_gt.append(pc)

    # -------------------------------------------------------
    # МОДЕЛЬ (создание облаков точек для предсказаний модели)
    # -------------------------------------------------------
    all_pointcloud_object_model = []
    # перебор всех объектов во всех установленных объектах
    for k_object_name, v_object_index in dict_to_use.items():
        # цикл по всем объектам, начиная с сегментированных точек, обнаруженных моделью
        for k_object_model, v_object_model in dict_of_tensors_allpoints_per_object_reduced.items():
            # Если нашли объект с меткой в каталоге dict_to_use - добавляем точки на 3d сцену
            if b_multiple_seg and k_object_model == k_object_name:
                if b_hide_wall and k_object_model == "wall":
                    continue
                pc_model = o3d.geometry.PointCloud()
                pc_model.points = o3d.utility.Vector3dVector(v_object_model[:, :3])  # получить координаты xyz
                pc_model.paint_uniform_color(vparams[k_object_name + '_color'])
                all_pointcloud_object_model.append(pc_model)
            elif k_object_name == k_object_model and k_object_model == vparams["str_object_to_visualize"]:
                pc_model = o3d.geometry.PointCloud()
                pc_model.points = o3d.utility.Vector3dVector(v_object_model[:, :3])  # получить координаты xyz
                pc_model.paint_uniform_color(vparams[k_object_name + '_color'])
                all_pointcloud_object_model.append(pc_model)

    # временная метка как ссылка на сохраненные файлы
    ts = calendar.timegm(time.gmtime())

    visualizer_gt = o3d.visualization.Visualizer()
    visualizer_gt.create_window(window_name='Segmentation GT id ' + vparams["str_object_to_visualize"])

    visualizer_model = o3d.visualization.Visualizer()
    visualizer_model.create_window(window_name='Segmentation MODEL id ' + vparams["str_object_to_visualize"])

    # -----------------------------------------
    # GT И ВИЗУАЛИЗАЦИ РЕЗУЛЬТАТОВ СЕГМЕНТАЦИИ
    # -----------------------------------------
    # КОМНАТА
    if b_show_room_points:
        # создать облако точек объекта
        pc_room = o3d.geometry.PointCloud()
        pc_room.points = o3d.utility.Vector3dVector(room_points_gt[:, :3])
        if draw_original_rgb_data:
            # окрасить комнату
            pc_room.colors = o3d.utility.Vector3dVector(room_points_gt[:, 3:6])
        else:
            # окрасить в чб цвета
            pc_room.paint_uniform_color(cparams['Grey'])

        # добавить облака точек
        visualizer_gt.add_geometry(pc_room)
        visualizer_model.add_geometry(pc_room)

    # добавить только сегментированные объекты из GT файла, которые были получены
    for segment_gt in all_pointcloud_object_gt:
        visualizer_gt.add_geometry(segment_gt)

    # ----------------------------
    # GT (камера)
    # ----------------------------
    ctr = visualizer_gt.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(parameters_camera)

    # save image
    visualizer_gt.poll_events()
    visualizer_gt.update_renderer()
    visualizer_gt.run()

    filename = get_file_name(ts, str_area_and_office, b_multiple_seg, True, "PV1", "_hidden_wall_")
    visualizer_gt.capture_screen_image(camera_folder + '/' + filename)

    # close window
    visualizer_gt.destroy_window()

    # add only the segmented objects from the GT file that are being studied
    for segment_model in all_pointcloud_object_model:
        visualizer_model.add_geometry(segment_model)

    # ---------------------
    # МОДЕЛЬ (камера)
    # ---------------------
    ctr = visualizer_model.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(parameters_camera)

    # save image
    visualizer_model.poll_events()
    visualizer_model.update_renderer()
    visualizer_model.run()

    filename = get_file_name(ts, str_area_and_office, b_multiple_seg, False, "PV1", "_hidden_wall_")
    visualizer_model.capture_screen_image(camera_folder + '/' + filename)

    # close window
    visualizer_model.destroy_window()


def get_file_name(timestamp,
                  str_area_and_office="",
                  b_multiple_seg=False,
                  b_is_GT_file=False,
                  str_PV_version="",
                  str_sufix_hidden_wall=""):
    if b_is_GT_file:

        if b_multiple_seg:
            return str(timestamp) + '__' + str(str_area_and_office) + "_" + \
                str(hparams["dimensions_per_object"]) + "_dims_" + \
                str(hparams["num_classes"]) + "_clases_" + str_sufix_hidden_wall + \
                "_seg_GT_" + str_PV_version + ".png"
        else:
            return str(timestamp) + '__' + str(str_area_and_office) + \
                str(hparams["dimensions_per_object"]) + "_dims_" + \
                str(hparams["num_classes"]) + "_clases_" + str_sufix_hidden_wall + \
                "_seg_" + str(vparams["str_object_to_visualize"]) + "_ " + \
                "_GT_" + str_PV_version + ".png"
    else:
        if b_multiple_seg:
            return str(timestamp) + '__' + str(str_area_and_office) + "_" + \
                str(hparams["num_points_per_room"]) + "_room_points_" + \
                str(hparams["dimensions_per_object"]) + "_dims_" + \
                str(hparams["num_classes"]) + "_clases_" + str_sufix_hidden_wall + \
                str(hparams["epochs"]) + "_epochs_seg_" + \
                "_MODEL_" + str_PV_version + ".png"
        else:
            return str(timestamp) + '__' + str(str_area_and_office) + \
                str(hparams["num_points_per_room"]) + "_room_points_" + \
                str(hparams["dimensions_per_object"]) + "_dims_" + \
                str(hparams["num_classes"]) + "_clases_" + \
                str(hparams["epochs"]) + "_epochs_" + str_sufix_hidden_wall + \
                str(vparams["str_object_to_visualize"]) + \
                "seg_" + \
                "_MODEL_" + str_PV_version + ".png"
