from settings import *


class TNet(nn.Module):
    """
    Блок T-Net для архитектуры PointNet

    Используется для преобразования как входного облака точек, так и вектора признаков

    T-Net выравнивает все входные данные по каноническому пространству перед извлечением признаков.
    Как это делается? Он предсказывает матрицу аффинного преобразования
    который будет применен к координатам входных точек (x, y, z).

    T-Net выравнивает все входные данные по каноническому пространству, изучая
    матрицу преобразования
    """
    def __init__(self, input_dim, output_dim):
        super(TNet, self).__init__()
        self.output_dim = output_dim

        # Conv1d для точечно-независимого выделения признаков
        # так как kernel_size = 1 -> аналогично FC-слою (nn.Linear)
        self.conv_1 = nn.Conv1d(input_dim, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 1024, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(1024)
        self.bn_4 = nn.BatchNorm1d(512)
        self.bn_5 = nn.BatchNorm1d(256)

        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, self.output_dim * self.output_dim)

    def forward(self, x):
        # x.shape[batch_size, num_points_per_object, xyzrgb]
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        # Теперь x имеет размерность [batch_size, input_dim - xyzrgb, num_points_per_object]
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        # Определяем Maxpool1D и применяем его напрямую к x
        #  Применение операции MaxPooling по всем точкам, результатом
        #  является один вектор размерности [batch_size, 1024, 1]
        x = nn.MaxPool1d(num_points, return_indices=False, ceil_mode=False)(x)
        # Преобразование данных к размерности [batch_size, 1024]
        x = x.view(-1, 1024)

        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        # Применение третьего полносвязного слоя,
        # результат - вектор размерности [batch_size, output_dim * output_dim]
        x = self.fc_3(x)

        # Создание единичной матрицы размерности output_dim x output_dim
        identity_matrix = torch.eye(self.output_dim).to(hparams['device'])
        # Преобразование вектора в матрицу размерности
        # [batch_size, output_dim, output_dim] и добавление единичной матрицы для начального приближения.
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix

        #  возвращает матрицу преобразования размерности [batch_size, output_dim, output_dim]
        return x


class BasePointNet(nn.Module):
    """
        Класс BasePointNet реализует базовую часть архитектуры PointNet,
        от входных данных до глобального вектора признаков.

        Он включает в себя два блока TNet для выравнивания входных
        данных и признаков, а также несколько сверточных слоев с
        нормализацией и активацией.

        Основная задача этого класса — извлечение и преобразование
        признаков для дальнейшего использования в задаче
        сегментации.
    """

    def __init__(self, point_dimension):
        super(BasePointNet, self).__init__()
        # TNet для преобразования входных данных
        self.input_transform = TNet(input_dim=point_dimension, output_dim=point_dimension)
        # TNet для преобразования признаков
        self.feature_transform = TNet(input_dim=64, output_dim=64)

        self.conv_1 = nn.Conv1d(point_dimension, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 1024, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        # x.shape([batch_size, num_points_per_room, dimensions_per_object]) [32, 4096, 6]
        num_points = x.shape[1]

        # Применение TNet для входных данных. Результатом является матрица
        # преобразования размерности [batch_size, dimensions_per_object, dimensions_per_object] (t-net tensor)
        input_transform = self.input_transform(x)

        #  Преобразование входных данных с помощью матрицы преобразования. Матричное умножение.
        #  Результат имеет размерность [batch_size, num_points_per_room, dimensions_per_object]
        x = torch.bmm(x, input_transform)

        # Транспонирование для удобства применения сверточных слоев.
        # Теперь x имеет размерность [batch_size, dimensions_per_object, num_points_per_room]
        x = x.transpose(2, 1)
        # Применение первого сверточного слоя, нормализация и функция активации ReLU.
        # Результат имеет размерность [batch_size, 64, num_points_per_room]
        x = F.relu(self.bn_1(self.conv_1(x)))
        # Транспонирование обратно к размерности [batch_size, num_points_per_room, 64]
        x = x.transpose(2, 1)

        # Применение TNet для признаков. Результатом является
        # матрица преобразования размерности [batch_size, 64, 64]
        feature_transform = self.feature_transform(x)
        # Преобразование признаков с помощью матрицы преобразования.
        # Результат имеет размерность [batch_size, num_points_per_room, 64]
        x = torch.bmm(x, feature_transform)

        # Сохранение локальных признаков для задач сегментации
        segmentation_local_features = x

        # Транспонирование для удобства применения сверточных слоев.
        # Теперь x имеет размерность [batch_size, 64, num_points_per_room]
        x = x.transpose(2, 1)
        #  Применение второго сверточного слоя, нормализация и функция активации ReLU.
        #  Результат имеет размерность [batch_size, 128, num_points_per_room]
        x = F.relu(self.bn_2(self.conv_2(x)))
        # Применение третьего сверточного слоя, нормализация и функция активации ReLU.
        # Результат имеет размерность [batch_size, 1024, num_points_per_room]
        x = F.relu(self.bn_3(self.conv_3(x)))

        # Применение Max-Pooling по всем точкам, результатом
        # является один вектор размерности [batch_size, 1024, 1]
        x, ix = nn.MaxPool1d(num_points, return_indices=True, ceil_mode=False)(x)

        # Преобразование данных к размерности [batch_size, 1024],
        # что является глобальным вектором признаков.
        global_feature_vector = x.view(-1, 1024)

        #  возвращает несколько значений:
        # global_feature_vector: Глобальный вектор признаков размерности [batch_size, 1024].
        # feature_transform: Матрица преобразования признаков размерности [batch_size, 64, 64].
        # segmentation_local_features: Локальные признаки для задач сегментации.
        return global_feature_vector, feature_transform, segmentation_local_features


class SegmentationPointNet(nn.Module):
    """

    Класс SegmentationPointNet реализует семантическую сегментационную
    сеть архитектуры PointNet.

    Этот класс выполняет сегментацию облака точек, объединяя локальные
    признаки точек с глобальными признаками, чтобы получить вероятность
    принадлежности каждой точки к определенному классу.
    """

    def __init__(self, num_classes, point_dimension=3):
        super(SegmentationPointNet, self).__init__()

        self.num_classes = num_classes
        # Экземпляр базового модуля PointNet, который извлекает глобальные и локальные признаки.
        self.base_pointnet = BasePointNet(point_dimension=point_dimension)

        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.num_classes, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # x.shape([batch_size, num_points_per_room, dimensions_per_object]])
        num_points = x.shape[1]

        # Вызов базовой сети PointNet для получения глобального вектора признаков,
        # матрицы преобразования признаков, выхода TNet, индексов Max-Pooling
        # и локальных признаков для сегментации
        # global_feature_vector.shape([batch_size, 1024])
        # seg_local_feats.shape([batch_size, num_points_per_room, 64])
        global_feature_vector, feature_transform, seg_local_feats = self.base_pointnet(x)

        # Изменение размерности глобального вектора признаков и
        # его повторение для каждого из точек в облаке точек.
        # Результат имеет размерность [batch_size, num_points_per_room, 1024]
        global_feature_vector = global_feature_vector.view(-1, 1, 1024).repeat(1, num_points, 1)

        #  Объединение локальных и глобальных признаков вдоль
        #  третьего измерения (каналов).
        #  Результат имеет размерность [batch_size, num_points_per_room, 1088]
        x = torch.cat((seg_local_feats, global_feature_vector), dim=2)

        #  Транспонирование для удобства применения сверточных слоев.
        #  Теперь x имеет размерность [batch_size, 1088, num_points_per_room].
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Применение четвертого сверточного слоя для получения
        # вероятностей классов. Результат имеет размерность
        # [batch_size, num_classes, num_points_per_room].
        x = self.conv4(x)

        # Применение функции log_softmax вдоль размерности классов,
        # чтобы получить логарифмические вероятности для каждого
        # класса. Результат имеет размерность [batch_size, num_classes, num_points_per_room].
        predictions = F.log_softmax(x, dim=1)

        # возвращает несколько значений:
        # predictions: Логарифмические вероятности классов для каждой точки,
        #        размерность [batch_size, num_classes, num_points_per_room].
        # feature_transform: Матрица преобразования признаков, размерность [batch_size, 64, 64].
        return predictions, feature_transform
