from settings import *


def create_dataloaders(dataset):
    """
    Создать dataloader'ы (загрузчики данных)
    """

    train_dataset = dataset[0]
    val_dataset = dataset[1]
    test_dataset = dataset[2]

    # Dataloaders creation
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        num_workers=hparams["num_workers"]
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        num_workers=hparams["num_workers"]
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        num_workers=hparams["num_workers"]
    )

    return train_dataloader, val_dataloader, test_dataloader


class PointSampler:
    """
    Служебный класс для уменьшения/увеличения (up-, downsampling - сэмплирования) дискретизации облака точек
    чтобы все облака точек имели одинаковый размер для работы dataloader'ов.

    Загрузчики данных требуют, чтобы все их входные точки были одинакового размера
    """

    def __init__(self, point_cloud, max_points):
        """
        Аргументы:
            point_cloud: torch файл
            max_points: Максимальное количество точек, которое должно быть в облаке точек
        """
        self.point_cloud = point_cloud
        self.max_points = max_points

    def sample(self):
        # Проверка, что облако точек уже тензор
        if not torch.is_tensor(self.point_cloud):
            self.point_cloud = torch.tensor(self.point_cloud)

        point_cloud_len = len(self.point_cloud)

        # Поскольку тензоры облака точек будут иметь разное количество точек в строке,
        # нам нужно установить общий размер для них всех
        # Torch Dataloaders ожидают, что каждый тензор будет иметь одинаковый размер
        if len(self.point_cloud) > self.max_points:
            # Дискретизация точек
            # Изменение индексации (torch.randperm(4) -> tensor([2, 1, 0, 3]))
            idxs = torch.randperm(len(self.point_cloud))[:self.max_points]
            self.point_cloud = self.point_cloud[idxs]
        else:
            # Интерполяция точек для увеличения
            padding_points = self.max_points - point_cloud_len
            existing_indices = torch.randint(0, point_cloud_len, (padding_points,))
            new_points = self.point_cloud[existing_indices] + torch.randn(padding_points,
                                                                          self.point_cloud.size(1)) * 0.01
            self.point_cloud = torch.cat((self.point_cloud, new_points), dim=0)

        return self.point_cloud

# ------------------------------------------------------------------------------
# ДАТАСЕТЫ ДЛЯ СЕГМЕНТАЦИИ
# ------------------------------------------------------------------------------


class S3DISDataset4SegmentationBase(torch.utils.data.Dataset):
    """
    Класс для создания датасетов S3DIS для сегментации.

    Наборы данных создаются из содержимого папки sliding_windows

    Элементы набора данных выбираются в зависимости от области, к которой они принадлежат:

    - Обучающий набор данных: Области 1, 2, 3 и 4
    - Набор данных для проверки: Область 5
    - Набор данных для тестирования: Область 6

    Из метода S3DIS_Summarized.create_sliding_windows() создается папка,
    содержащая все предварительно обработанные скользящие блоки
    для всех пространств/комнат (spaces) в наборе данных.

    Гайд по наименованию:

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

    def __init__(self, root_dir, all_objects_dict, transform=None, proper_area=None):
        """
        Аргументы:
            root_dir (строка): Каталог со всеми предварительно обработанными скользящими окнами
            all_objects_dict: каталог, содержащий идентификатор
                сопоставляемого object_ID <-> object_name
            transform (вызываемый, необязательный): Необязательное преобразование,
                которое будет применено к образцу.
            proper_area (список): Области, которые будут использоваться для создания соответствующего набора данных
                        Области 1, 2, 3 и 4 для обучения
                        Область 5 для проверки
                        Область 6 для тестирования
        """

        self.root_dir = root_dir
        self.transform = transform
        self.all_objects_dict = all_objects_dict
        self.proper_area = proper_area
        self.number_of_points = 0

        # Получить отсортированный список скользящих окон
        # Получить только файлы Pytorch, если существуют другие типы файлов
        self.all_sliding_windows = sorted(
            [f for f in os.listdir(path_to_current_sliding_windows_folder) if ".pt" in f])

        self.sliding_windows = []

        # Установка скользящих оков в соответствии с областью и назначением
        # для train: Зона 1, Зона 2, зона 3 и зона 4
        # для val: Зона 5
        # для test: Зона 6
        for area in self.proper_area:
           for f in self.all_sliding_windows:
                if f.startswith(area):
                    self.sliding_windows.append(f)

    def __len__(self):
        return len(self.sliding_windows)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Получение имени скользящего окна
        sliding_window_name = self.sliding_windows[idx]

        # Получить путь к скользящему окну
        path_to_sliding_window_file = os.path.join(
            path_to_current_sliding_windows_folder,
            sliding_window_name)

        # Открыть файл скользящего окна (они сохранены как torch файлы)
        sliding_window = torch.load(path_to_sliding_window_file,
                                    map_location=torch.device(hparams["device"]))

        # Сэмплирование точек, чтобы все элементы имели одинаковый размер для работы dataloader'а
        sliding_window = PointSampler(sliding_window, hparams['num_points_per_room']).sample()

        # Количество колонок, которые будут возвращены для точки комнаты, будет зависеть от двух факторов:
        # 1) Учитываем ли мы цвет? (Нет: 3 колонок | Да: 6 колонок)
        # 2) Визуализируем ли мы отдельную комнату? (Нет: 3 столбца | Да: 6 столбцов)
        # Если мы визуализируем, нам нужно взять (x_rel, y_rel, z_rel) и (x y z)
        # (xyz) будут сохранены для построения графика из исходной комнаты, если они соответствуют объекту
        # Столбцы скользящего окна: (x_rel y_rel z_rel R G B x y z win_ID label)
        # - 3 относительные нормализованные точки (x_rel, y_rel, z_rel)
        # - 3 цвета (RGB)
        # - 3 абсолютные координаты (x, y, z)
        # - 1 идентификатор скользящего окна (win_ID)
        # - 1 метка точки для этой точки (label)

        # Срез из тензора
        # [start_row_index:end_row_index, start_column_index:end_column_index]
        sliding_window_points = sliding_window[:, :hparams["dimensions_per_object"]]

        point_labels = sliding_window[:, -1]

        return sliding_window_points, point_labels

    def __str__(self) -> str:
        msg_list = [100 * "-", "S3DIS DATASET INFORMATION ({})".format(self.__class__.__name__), 100 * "-",
                    "Summary file: {}".format(eparams['s3dis_summary_file']),
                    "Data source folder: {} ".format(self.root_dir), "Chosen areas: {} ".format(self.proper_area),
                    "Total points (from sliding windows): {} ".format(self.total_number_of_points),
                    "Total points (after sampling sliding windows at {} points/room): {} ".format(
                        hparams["num_points_per_room"],
                        round(self.total_number_of_points / hparams["num_points_per_room"])),
                    "Total dataset elements: {} (from a grand total of {})".format(
                        len(self.sliding_windows),
                        len(self.all_sliding_windows))]
        if not self.subset:
            # Создать dict, чтобы узнать, какие файлы скользящего окна относятся к каждой области
            sliding_windows_per_area = dict()
            for area in self.proper_area:
                sliding_windows_per_area[area] = [f for f in self.sliding_windows if f.startswith(area)]

            for k, v in sliding_windows_per_area.items():
                msg_list.append("From {} : {} ({}...{})".format(k, len(v), v[:3], v[-3:]))

        msg = '\n'.join(msg_list)
        msg += "\n"

        return str(msg)


class S3DISDataset4SegmentationTrain(S3DISDataset4SegmentationBase):
    """
    Дополненный класс S3DISDataset4SegmentationBase для создания набора данных, используемого
    для обучения
    """

    def __init__(self, root_dir, all_objects_dict, transform=None, proper_area=None):
        S3DISDataset4SegmentationBase.__init__(self, root_dir, all_objects_dict, transform=None,
                                               proper_area=training_areas)


class S3DISDataset4SegmentationVal(S3DISDataset4SegmentationBase):
    """
    Дополненный класс S3DISDataset4SegmentationBase для создания набора данных, используемого
    для валидации (проверки)
    """

    def __init__(self, root_dir, all_objects_dict, transform=None, proper_area=None):
        S3DISDataset4SegmentationBase.__init__(self, root_dir, all_objects_dict, transform=None, proper_area=val_areas)


class S3DISDataset4SegmentationTest(S3DISDataset4SegmentationBase):
    """
    Дополненный класс S3DISDataset4SegmentationBase для создания набора данных, используемого
    для тестирования
    """

    def __init__(self, root_dir, all_objects_dict, transform=None, proper_area=None):
        S3DISDataset4SegmentationBase.__init__(self, root_dir, all_objects_dict, transform=None, proper_area=test_areas)
