from settings import *


class S3DIS_Summarizer:
    """
    Класс для генерации ground truth файла.
    Также получает доп. информацию из датасета S3DIS, например, количество точек на объект
    """

    # Названия колонок, которые сохраняются в CSV файле после обхода папки с датасетом
    S3DIS_summary_cols = [
        "Area",
        "Space",
        "Space_Label",
        "Space_ID",
        "Object",
        "Object_Points",
        "Object_Label",
        "Object_ID"
    ]

    def __init__(self, path_to_data):
        """
        Обход датасета с целью поиска информации о:
            - Areas - Области
            - Spaces (Rooms) - Комнаты (Пространства)
            - Objects (Table itc.) - Объекты
            - Points per object - Количество точек на объект
            - Labels (for Areas and Spaces) - Лейблы

        Структура датасета S3DIS:
        Path_to_data\\Area_1\\space_X
                            \\space_Y
                            \\space_Z\\Annotations\\object_1
                            \\space_Z\\Annotations\\object_2
        

        Входные данные:
            - Path to dataset

        Выходные данные: CSV файл, содержащий следующие колонки:
            - Area
            - Space
            - Space_label
            - Space_ID
            - Object
            - Points per object
            - Object_label
            - Object_ID

        Аргументы:
            - path_to_data (string): путь к датасету
        """
        self.path_to_data = path_to_data
        self.path_to_summary_file = os.path.join(self.path_to_data, eparams['s3dis_summary_file'])

        # Не выполнять поиск информации, если CSV файл существует
        if os.path.exists(self.path_to_summary_file):
            # Создать Pandas DataFrame, чтобы сохранить в полях класса
            self.summary_df = pd.read_csv(self.path_to_summary_file,
                                          header=0,
                                          usecols=self.S3DIS_summary_cols,
                                          sep="\t")
            with open(os.path.join(self.path_to_data, eparams['spaces_file'])) as json_file:
                self.spaces_dict = json.load(json_file)
            with open(os.path.join(self.path_to_data, eparams['objects_file'])) as json_file:
                self.objects_dict = json.load(json_file)
            return

        print("Создание файла истины на основе обхода папок {} in {}".format(
            eparams['s3dis_summary_file'], self.path_to_data))

        # Каждя строка S3DIS CSV файла содержит:
        # (area, space, space_label, space_id, object,
        # points_per_object, object label, object_id, health_status)
        summary_line = []

        # Доп. переменные (dict) для присвоения комнатам и объектам идентификатора
        self.spaces_dict = dict()
        total_spaces = 0
        self.objects_dict = dict()
        total_objects = 0

        # Оставляем только папки (папки для areas), назнания которых начинаются на Area_XXX
        areas = dict((folder, '') for folder in sorted(os.listdir(self.path_to_data)) if folder.startswith('Area'))

        # Для каждой папки области (area) рассматриваем комнаты (spaces), которые там есть
        for area in areas:

            # os.path.join берет сепаратор, характерный для данной ОС ("/", "\")
            path_to_spaces = os.path.join(self.path_to_data, area)

            # Берем только комнаты (spaces) для каждой area, убирая мета-файлы (".DStore")
            spaces = sorted([space for space in os.listdir(path_to_spaces)
                             if not '.' in space])

            # Для каждого пространства space (комнаты) берем объекты в нем находящиеся
            for space in spaces:
                path_to_objects = os.path.join(path_to_spaces, space, "Annotations")

                # Получаем лейбл комнаты (space)
                # Из hallway_1, hallway_2 оставляем только "hallway"
                space_label = space.split("_")[0]

                # Обновляем словарь комнат и их порядковых номеров
                # {'hall': 0, 'wall': 1, ...}
                if space_label not in self.spaces_dict:
                    self.spaces_dict[space_label] = total_spaces
                    total_spaces += 1

                space_idx = self.spaces_dict[space_label]

                # Получаем список файлов с объектами
                objects = sorted([object for object in os.listdir(path_to_objects)
                                  if object.split("_")[0] in objects_set])

                if objects:
                    desc = "Получение точек из объектов {} {}".format(area, space)

                    for object in tqdm(objects, desc=desc):
                        # Получаем объект
                        # Из chair_1, chair_2 остается только "chair"
                        object_label = object.split("_")[0]

                        # Обновляем словарь с объектами и их порядковыми номерами
                        # {'chair': 0, 'table': 1, ...}
                        if object_label not in self.objects_dict:
                            self.objects_dict[object_label] = total_objects
                            total_objects += 1

                        object_idx = self.objects_dict[object_label]

                        # Получаем количество точек на объект
                        with open(os.path.join(path_to_objects, object)) as f:
                            points_per_object = len(list(f))

                        # Сохраняем рез-ат обхода в файл:
                        # (Area, space, space_label, space_ID, object,
                        # points_per_object, object_label, object_ID)
                        summary_line.append((area, space, space_label, space_idx, object,
                                             points_per_object, object_label, object_idx))

        # Сохраняем данные в CSV файл
        self.summary_df = pd.DataFrame(summary_line)
        self.summary_df.columns = self.S3DIS_summary_cols
        self.summary_df.to_csv(os.path.join(self.path_to_data, eparams['s3dis_summary_file']), index=False,
                               sep="\t")

        # сохраняем комнаты (spaces) и объекты (objects) в json файлы
        with open(os.path.join(self.path_to_data, eparams['spaces_file']), 'w', encoding="utf-8") as file:
            json.dump(self.spaces_dict, file)

        with open(os.path.join(self.path_to_data, eparams['objects_file']), 'w', encoding="utf-8") as file:
            json.dump(self.objects_dict, file)

    def label_points_for_semantic_segmentation(self):
        """
        Создание единого аннотированного файла (для room/space)

        Метод outlook:

         - Метка, которая будет присвоена каждой точке в облаке, будет основана на имени файла,
            в котором находится точка (например, все точки в chair_1.txt будут иметь метку "chair").
            Фактически, поскольку уже созданный сводный CSV файл уже содержит эту информацию,
            он будет использоваться для получения соответствующей метки для каждой точки

         - Для каждого пространства/комнаты (space) будет создан новый файл, содержащий все
            точки с аннотациями. Этот файл будет называться space_annotated.txt
            (например, conferenceRoom_1_annotated.txt) и будет сохранен рядом с исходным space(комнатой)
            без аннотаций (например, Area_1\\office_1\\office_1_annotated.txt)

         - Этот новый файл:
            - будет представлять собой объединение всех файлов в папке "Annotations"
                (поскольку файл Area_1\\office_1\\office_1.txt представляет собой
                сумму всех файлов в папке Area_1\\office_1\\Annotations).
              - будет добавлен дополнительный столбец для метки
                (на основе имени файла, которое мы объединяем)

        Следуя примеру Area_1\\office_1, будет создан новый файл с именем
        Area_1\\office_1\\office_1_annotated.txt.
        Этот файл будет содержать дополнительный столбец для размещения метки для
        каждой точки в облаке точек.

        уникальные комбинации area-space
        unique_area_space_df (272 строки)

        Area_space:
                Area             Space
        0    Area_1              WC_1
        1    Area_1  conferenceRoom_1
        2    Area_1  conferenceRoom_2
        3    Area_1        copyRoom_1
        4    Area_1         hallway_1
        ...     ...               ...
        267  Area_6          office_7
        268  Area_6          office_8
        269  Area_6          office_9
        270  Area_6       openspace_1
        271  Area_6          pantry_1
        """

        # Получение уникальных комбинаци area-space из summary_df
        # чытобы узнать точное количество комнат, которые необходимо разметить (272)
        unique_area_space_df = self.summary_df[["Area", "Space"]].drop_duplicates(ignore_index=True)

        # Определение панели процессов, которая будет отображаться при обработке файлов
        progress_bar = tqdm(unique_area_space_df.iterrows(), total=len(unique_area_space_df))

        # Начало обработки файлов
        print("Создание размеченного файла для комнаты (из файла: {})".format(eparams['s3dis_summary_file']))
        print("Время начала: ", datetime.now())

        for (idx, row) in progress_bar:
            # Получение текущих area и space
            area = row["Area"]
            space = row["Space"]

            # Определение пути к папке, в которой будет сохранен размеченный файл
            # (например, Area_1\office_1\office_1_annotated.txt)
            path_to_space = os.path.join(self.path_to_data, area, space)
            path_to_objs = os.path.join(path_to_space, "Annotations")
            sem_seg_file = space + eparams['pc_file_extension_sem_seg_suffix'] + eparams['pc_file_extension']
            path_to_sem_seg_file = os.path.join(path_to_space, sem_seg_file)

            # Если файл уже существует - пропуск
            if os.path.exists(path_to_sem_seg_file):
                msg = "Обработка файла {}_{}: Пропущен".format(area, space)
                progress_bar.set_description(msg)

            # Создание размеченного файла
            else:
                # Получение всех объектов, которые находятся на текущей area и space
                # (NOTE: из the summary_df!)
                objects_df = self.summary_df[(self.summary_df["Area"] == area) &
                                             (self.summary_df["Space"] == space)]

                # Чтение каждого object/class файла
                # idx начинается с 0 до len(objects_df.index)
                # i представляет индекс из summary_df (41, 73,...)
                for idx, i in enumerate(objects_df.index):

                    # GПолучение инф-ии линия за линией
                    summary_line = self.summary_df.iloc[i]

                    # Получение инф-ии об объекте на каждой строке файла
                    obj_file = summary_line["Object"]
                    obj_label_id = summary_line["Object_ID"]

                    if "clutter" not in obj_file:
                        # Обновление сообщения на progress bar
                        msg = "Обработка файла {}_{}: {}/{} ({})".format(area,
                                                                         space, idx + 1, len(objects_df), obj_file)
                        progress_bar.set_description(msg)

                        # Открыть файл как Numpy array
                        path_to_obj = os.path.join(path_to_objs, obj_file)
                        obj_data = np.genfromtxt(path_to_obj,
                                                 delimiter=' ',
                                                 names=None)

                        # Создание вектора столбца, копирующего object_label_id столько раз,
                        # сколько строк в данных объекта
                        label_col = np.array([obj_label_id for _ in obj_data])

                        # Конкатенация лейбла со всеми точками объекта
                        sem_seg_data = np.column_stack((obj_data, label_col))

                        # Сохранение файла с разметкой как Numpy txt файла
                        with open(path_to_sem_seg_file, 'a') as f:
                            np.savetxt(f, sem_seg_data, fmt="%4.3f")

        progress_bar.close()
        print("Время завершения: ", datetime.now())

    def get_labels(self):
        """
        Создание словарей с различными простраснтвами/spaces (conf rooms, hall ways,...)
        и объектами/objects (table, chairs,...) в Area

        Output:
            - Правильный dict, содержашщий объекты, с которыми работаем
        """

        if not os.path.exists(self.path_to_summary_file):
            msg = "Итоговный файл S3DIS {} не найден {}."
            msg += "Итоговый файл будет сгенерирован автоматически"
            print(msg.format(eparams['s3dis_summary_file'], self.path_to_data))
            self.__init__(self.path_to_data)

        return self.objects_dict

    def create_sliding_windows(self, rebuild=False):
        """
        Создает файлы, в которых будут храниться скользящие окна

        Параметры скользящих окон задаются юзером и хранятся в settings.py.
        Эти параметры:

        w: ширина скользящего окна (sliding window)
        d: глубина скользящего окна (sliding window)
        h: высота скользящего окна (sliding window)
        o: перекрытие последовательного скользящего окна (sliding window)

        Все скользящие окна создаются путем разделения ранее созданного
        файла комнаты с аннотациями (\\Area_N\\Space_X\\space_x_annotated.txt)
        в соответствии с параметрами, определенными юзером.

        Чтобы упростить управление набором данных, все скользящие окна для всех
        доступных комнат сохраняются в одной папке
        и будут сохранены в виде тензоров Pytorch:

        Area_N
        sliding_windows
        ├── w_X_d_Y_h_Z_o_T
            ├── Area_N_Space_J_winK.pt

        где:
        w_X: ширина скользящего окна (sliding window)
        d_Y: глубина скользящего окна (sliding window)
        h_Z: высота скользящего окна (sliding window)
        o_T: перекрытие последовательного скользящего окна (sliding window)
        winK: последовательный ID скользящего окна
        """

        # Удаление существующих скользящих окон, если необходимо
        if rebuild:
            print("Удаление содержимого папки ", path_to_current_sliding_windows_folder)
            for f in tqdm(os.listdir(path_to_current_sliding_windows_folder)):
                os.remove(os.path.join(path_to_current_sliding_windows_folder, f))

        # Если файлы действительно существуют, пропустите создание скользящих окон
        # Внимание: Если процесс создания скользящего окна каким-либо образом прервется,
        # файлы torch придется удалить вручную
        if len([f for f in os.listdir(path_to_current_sliding_windows_folder) if ".pt" in f]) != 0:
            return

        # Получение уникальных комбинаци area-space из summary_df
        # чтобы узнать точное количество комнат (272)
        unique_area_space_df = self.summary_df[["Area", "Space"]].drop_duplicates()

        # Создание скользящих окон для каждого уникального сочетания area-space
        print("Создание скользящих окон для:")
        progress_bar = tqdm(unique_area_space_df.iterrows(), total=len(unique_area_space_df))

        for (idx, row) in progress_bar:
            # Получить текущие area и space
            area = row["Area"]
            space = row["Space"]

            # Обновление инф-ии в progess bar
            progress_bar.set_description(area + "_" + space)

            # Создание скользящего окна
            self.create_sliding_windows_for_a_single_room(area, space, path_to_current_sliding_windows_folder)

    def create_sliding_windows_for_a_single_room(self, area, space, folder):
        """
        Создание скользящих окон для одного сочетания area-space
        """

        # Для удобства укажем параметры скользящих окон в локальных переменных
        win_width = hparams['win_width']
        win_depth = hparams['win_depth']
        win_height = hparams['win_height']
        overlap = hparams['overlap']
        overlap_fc = 100 - overlap
        win_fill = hparams['window_filling']

        # Открываем файл аннотаций для комнаты
        # (наприме, Area_1\office_1\office_1_annotated.txt)
        sem_seg_file = space + eparams["pc_file_extension_sem_seg_suffix"] + eparams["pc_file_extension"]
        path_to_space = os.path.join(self.path_to_data, area, space)
        path_to_room_annotated_file = os.path.join(path_to_space, sem_seg_file)

        data = np.genfromtxt(path_to_room_annotated_file,
                             dtype=float,
                             skip_header=1,
                             delimiter='',
                             names=None)

        # Получаем массивы данных и меток
        # При создании раздвижных окон необходимо учитывать цвет
        data_points = data[:, :6]
        point_labels = data[:, -1]

        # Создание векторов для X, Y, Z координат
        abs_x = data_points[:, 0]
        abs_y = data_points[:, 1]
        abs_z = data_points[:, 2]

        '''
        # FOR DEBUGGING PLOT X_Y ROOM SCATTER
                fig = plt.figure()
                ax = fig.add_subplot()
                ax.scatter(tri_points[:,0].tolist(), tri_points[:,1].tolist())
        '''

        # Найдем roommax_x, roommax_y, roommin_x, roommin_y, roommin_z из всех точек комнаты.
        # Началом координат будет (roomin_x, roommin_y, roommin_z)
        roommax_x = max(abs_x)
        roommax_y = max(abs_y)
        roommin_x = min(abs_x)
        roommin_y = min(abs_y)
        roommin_z = min(abs_z)

        # Переменными одного окна являются 4 угла в системе X-Y.
        # Они определяются комбинациями x и y их max и началом координат и
        # min значениями каждого окна:
        #   - (winmin_x winmin_y) --> начало координат окна
        #   - winmax_x = winmin_x + win_width --> макс значения x
        #   - winmax_y = winmin_y + win_depth --> макс значения y

        # Перемещаем окно по x до тех пор пока winmax_x>roommax_x
        # и по y до тех пор пока winmax_y>roommax_y

        # Определяем векторы координат
        # winmax_z определяется, но не используется, тк не беспокоимся о высоте
        # (Используем все точки по Z)
        # np.arange возвращает равномерно распределенные значения в пределах заданного интервала
        # np.arange(start, stop, step)
        winmin_xvec = np.arange(roommin_x, roommax_x, overlap_fc / 100 * win_width)
        winmin_yvec = np.arange(roommin_y, roommax_y, overlap_fc / 100 * win_depth)
        winmin_z = roommin_z
        winmax_z = roommin_z + win_height

        # ID - номер окна с более чем 0 точек в комнат,
        # необходим, чтобы отделять одно окно от другого
        win_count = 0

        # Для каждого возможного начала координат каждого окна в комнате найдем точки,
        # "запертые" внутри него, и преобразуем их в относительные нормализованные координаты
        # itertools.product('ABCD', repeat=2) возвращает:
        # AA AB AC AD BA BB BC BD CA CB CC CD DA DB DC DD
        for (winmin_x, winmin_y) in itertools.product(winmin_xvec, winmin_yvec):

            # Найдем макс значения x и y в данном окне
            winmax_x = winmin_x + win_width
            winmax_y = winmin_y + win_depth

            # Получаем облако точек всего помещения, из которого мы будем выбирать точки окон
            tri_points_aux = data_points
            labels_aux = point_labels

            # Выбераем только те точки, которые находятся в пределах заданных значений x для конкретного окна
            # point_sel - это матрица True/False
            point_sel = np.array((tri_points_aux[:, 0] > winmin_x) & (tri_points_aux[:, 0] < winmax_x))
            tri_points_aux = tri_points_aux[point_sel, :]
            labels_aux = labels_aux[point_sel]

            # Выбераем только те точки, которые находятся в пределах заданных значений y для конкретного окна
            point_sel = np.array((tri_points_aux[:, 1] > winmin_y) & (tri_points_aux[:, 1] < winmax_y))
            tri_points_aux = np.array(tri_points_aux[point_sel])
            labels_aux = labels_aux[point_sel]

            # Если нет точек в окне - игнорируем окно
            if tri_points_aux.size != 0:
                pcminx = min(tri_points_aux[:, 0])
                pcmaxx = max(tri_points_aux[:, 0])
                pcminy = min(tri_points_aux[:, 1])
                pcmaxy = max(tri_points_aux[:, 1])

                distance_x = pcmaxx - pcminx
                distance_y = pcmaxy - pcminy

                if (distance_x > win_fill * win_width and distance_y > win_fill * win_depth):
                    # tri_point_aux - матрица, которая содержит 3D точки
                    # внутри окна находятся абсолютные координаты
                    # Возьмем каждый вектор по-отдельности
                    abs_x_win = tri_points_aux[:, 0]
                    abs_y_win = tri_points_aux[:, 1]
                    abs_z_win = tri_points_aux[:, 2]

                    # Преобразуем координаты в относительные (относительно начала координат окна,
                    # а не абсолютного) и нормализуем по win_width, win_depth и win_height
                    # rel_x, rel_y, rel_z - вектора
                    rel_x = (abs_x_win - winmin_x) / win_width
                    rel_y = (abs_y_win - winmin_y) / win_depth
                    rel_z = (abs_z_win - winmin_z) / win_height

                    tri_points_rel = np.copy(tri_points_aux)

                    # Поместим относительные и нормализованные координаты в матрицу с информацией о цвете (rgb)
                    # tri_points aux - матрица с относительными координатами и инф-ии о цвете (rgb)
                    tri_points_rel[:, 0] = rel_x
                    tri_points_rel[:, 1] = rel_y
                    tri_points_rel[:, 2] = rel_z

                    # Конвертируем в 1D массив
                    labels_aux.shape = (len(labels_aux), 1)

                    # Создаем матрицу с:
                    # - 3 относительными нормализованными координатами
                    # - 3 цветами (rgb)
                    # - 3 абсолютными координатами
                    # - 1 ID окна
                    # - 1 лейбл объекта
                    tri_points_out = np.concatenate(
                        (tri_points_rel, tri_points_aux[:, 0:3], np.full((len(rel_x), 1), win_count), labels_aux),
                        axis=1)

                    # Конвертируем в torch тензор
                    tri_points_out = torch.from_numpy(tri_points_out).float()

                    # Сохраним тензор как файл
                    # Общим соглашением PyTorch является сохранение тензоров
                    # с использованием расширения файла .pt
                    sliding_window_name = area + '_' + space + "_"
                    sliding_window_name += "win" + str(win_count) + ".pt"
                    torch.save(tri_points_out, os.path.join(folder, sliding_window_name))

                    # Обновление ID скользящего окна
                    win_count += 1
