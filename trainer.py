from settings import *
import visualizer


class Trainer:
    def __init__(self, model, data_loader, objects_dict):
        self.test_and_val_dataloader = None
        self.model = model
        self.data_loader = data_loader
        self.train_data_loader = data_loader[0]
        self.val_data_loader = data_loader[1]
        self.optimizer = optim.Adam(self.model.parameters(), lr=hparams['learning_rate'],
                                    weight_decay=hparams['regularization_weight_decay_lambda'])
        self.epochs = hparams['epochs']
        self.objects_dict = objects_dict

    def compute_metrics(self, y_gt, y_preds):
        """
        Вычисление матрицы ошибок и других метрик, чтобы оценить эффективность модели.

        Аргументы:
        - y_gt: Вектор ground truth (истинных) лейблов
        - y_preds: Вектор, содержащий предсказание модели

        Возвращает:
            - F1 Score (Macro),
            - F1 Score (Micro),
            - F1 Score (Weighted),
            - Intersection over Union (IoU) (Macro)
            - Intersection over Union (IoU) (Micro)
            - Intersection over Union (IoU) (Weighted)
        """
        # Замена чисел в текст
        # Создание обратного dict, чтобы ускорить процесс замены чисел
        reverse_objects_dict = dict()
        for k, v in self.objects_dict.items():
            reverse_objects_dict[v] = k

        y_preds_text = []
        y_gt_text = []
        for n in y_preds:
            y_preds_text.append(reverse_objects_dict[n])

        for n in y_gt:
            # Предсказанания - int, истиные лейблы - float, необходимо перейти к int
            y_gt_text.append(reverse_objects_dict[int(n)])

        # Compute confusion matrix
        cm = confusion_matrix(y_gt_text, y_preds_text, labels=[k for k in self.objects_dict])

        # Get other metrics
        precision, recall, fscore, support = precision_recall_fscore_support(y_gt_text, y_preds_text,
                                                                             labels=[k for k in self.objects_dict])

        # Print the table
        cm_table = PrettyTable()
        per_object_scores = PrettyTable()
        avg_scores = PrettyTable()

        cm_table.field_names = ["Объект"] + [k for k in self.objects_dict]
        for idx, row in enumerate(cm.tolist()):
            row = [reverse_objects_dict[idx]] + row
            cm_table.add_row(row)
        print("\nМатрица ошибок")
        print(cm_table)
        print("")

        # Per object Precision, Recall and F1 scores
        print("\nОценки (на каждый объект)")
        per_object_scores.field_names = ["Оценки"] + [k for k in self.objects_dict]
        per_object_scores.add_row(["Precision"] + ["{:.4f}".format(v) for v in precision.tolist()])
        per_object_scores.add_row(["Recall"] + ["{:.4f}".format(v) for v in recall.tolist()])
        per_object_scores.add_row(["F1 Score"] + ["{:.4f}".format(v) for v in fscore.tolist()])
        print(per_object_scores)

        # Average scores
        # 1.- F1
        print("\nОценки (средние значения)")
        f1_score_macro = f1_score(y_gt_text, y_preds_text, average='macro')
        f1_score_micro = f1_score(y_gt_text, y_preds_text, average='micro')
        f1_score_weighted = f1_score(y_gt_text, y_preds_text, average='weighted')

        # 2.-Intersection over union (IoU)
        iou_score_macro = jaccard_score(y_gt_text, y_preds_text, average="macro")
        iou_score_micro = jaccard_score(y_gt_text, y_preds_text, average="micro")
        iou_score_weighted = jaccard_score(y_gt_text, y_preds_text, average="weighted")

        avg_scores.field_names = ["Оценка", "Macro", "Micro", "Weighted"]
        avg_scores.add_row(["F1", "{:.4f}".format(f1_score_macro), "{:.4f}".format(f1_score_micro),
                            "{:.4f}".format(f1_score_weighted)])
        avg_scores.add_row(["IoU", "{:.4f}".format(iou_score_macro), "{:.4f}".format(iou_score_micro),
                            "{:.4f}".format(iou_score_weighted)])
        print(avg_scores)
        print("")

        # From: https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)
        print("Общая точность: {}\n".format(ACC))

        return (f1_score_macro, f1_score_micro, f1_score_weighted,
                iou_score_macro, iou_score_micro, iou_score_weighted)

    def process_single_epoch(self, epoch):
        """
        Выполнение процесса тренировки одной эпохи:
        """
        # Списки для результатов (потери, точность,
        # истиные метки и метки предсказаний) выполнения тренировки за одну эпоху
        epoch_loss = []
        epoch_acc = []
        epoch_y_ground_truth = []
        epoch_y_predictions = []

        tqdm_desc = "Выполнение процесса обработки эпохи ({}/{})".format(epoch, self.epochs)
        for data in tqdm(self.train_data_loader, desc=tqdm_desc):
            # данные из dataloader'а: облако точек и метки
            points, labels = data

            # обнуление градиентов
            self.optimizer.zero_grad()
            # Прогон данных через модель и получение результатов
            predictions, feature_transform = self.model(points)
            # Создание единичной матрицы
            identity = torch.eye(feature_transform.shape[-1]).to(hparams['device'])

            # Lreg - регуляризация
            # Согласно оригинальной статье, регуляризацию следует применять только
            # при выравнивании пространства объектов (с большей
            # размерностью (64), чем матрица пространственного
            # преобразования (3)). С вычислением функции потерь
            # регуляризации оптимизация модели становится более стабильной
            # и обеспечивает лучшую производительность.
            # Потеря при регуляризации (с весом 0,001) добавляется к потере
            # при классификации softmax, чтобы приблизить матрицу к
            # ортогональной (цитируется по дополнительной информации из оригинальной статьи).

            reg_loss = torch.norm(
                identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1)))

            # Loss - Функция потерь:
            # Потеря с отрицательной логарифмической вероятностью
            # ----Для сегментации:
            # (N: batch_size; C: num_classes)
            # входные данные (прогнозы)– (N,C, d1, d2,...,dk))
            # цель - (N, d1, d2,...,dk)
            # Shape для предсказаний должен быть:
            # preds.shape[batch_size, num_classes, max_points_per_room]

            # По умолчанию она возвращает средневзвешенное значение выходных данных
            # https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
            loss = F.nll_loss(predictions, labels.long()) + 0.001 * reg_loss

            epoch_loss.append(loss.cpu().item())

            loss.backward()

            self.optimizer.step()

            # Из измерения num_classes (dim =1) найдем максимальное значение
            # max() возвращает кортеж (max_value, idx_of_the_max_value)
            # Возьмем индекс максимального значения, поскольку
            # сегментация объектов по классу основана на положении максимального значения
            # https://pytorch.org/docs/stable/generated/torch.max.html
            predictions = predictions.data.max(dim=1)[1]
            corrects = predictions.eq(labels.data).sum()
            accuracy = corrects.item() / predictions.numel()
            epoch_acc.append(accuracy)

            # Подготовка данных для confusion matrix
            labels = labels.view(-1, labels.numel()).squeeze(dim=0).tolist()
            predictions = predictions.view(-1, predictions.numel()).squeeze(dim=0).tolist()
            epoch_y_ground_truth.extend(labels)
            epoch_y_predictions.extend(predictions)

        print("\nЗначение функции потерь на test: ", round(np.mean(epoch_loss), 4))
        print("Точность на test:", round(np.mean(epoch_acc), 4))
        self.compute_val_acc()

        f1_scores = self.compute_metrics(epoch_y_ground_truth, epoch_y_predictions)

        return epoch_y_ground_truth, epoch_y_predictions, epoch_loss, epoch_acc, f1_scores

    def compute_val_acc(self):
        """
        """
        loss_val = []
        acc_val = []
        print("Валидация...")
        for data in tqdm(self.val_data_loader, ):
            points, labels = data

            predictions, feature_transform = self.model(points)
            identity = torch.eye(feature_transform.shape[-1]).to(hparams['device'])
            reg_loss = torch.norm(
                identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1)))

            loss = F.nll_loss(predictions, labels.long()) + 0.001 * reg_loss
            loss_val.append(loss.cpu().item())

            predictions = predictions.data.max(dim=1)[1]
            corrects = predictions.eq(labels.data).sum()
            accuracy = corrects.item() / predictions.numel()
            acc_val.append(accuracy)

        print("\nЗначение функции потерь на val: ", round(np.mean(loss_val), 4))
        print("Точность на val:", round(np.mean(acc_val), 4))
        gc.collect()

    def train_model(self):
        total_y_gt = []
        total_y_preds = []
        total_loss = []
        total_acc = []
        time_per_epoch = []
        # Установим макс значение функции потерь = бесконечности
        best_loss = np.inf
        for epoch in range(1, self.epochs + 1):
            # Время
            epoch_start_time = datetime.now()

            # Запуск одного шага тренировки модели (одна эпоха)
            scores = self.process_single_epoch(epoch)

            # Сохранить время (в сек.)
            epoch_end_time = datetime.now()
            time_per_epoch.append((epoch_end_time - epoch_start_time).seconds)

            # Сохранение результатов одной тренировки
            total_y_gt.extend(scores[0])
            total_y_preds.extend(scores[1])
            total_loss.extend(scores[2])
            total_acc.extend(scores[3])
            f1_score_macro, f1_score_micro, f1_score_weighted, iou_macro, iou_micro, iou_weighted = scores[4]

            # Сохраняем модель
            if (np.mean(total_loss) < best_loss):
                state = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }
                torch.save(
                    state,
                    os.path.join(eparams['pc_data_path'],
                                 eparams['checkpoints_folder'],
                                 eparams['checkpoint_name'])
                )
                best_loss = np.mean(total_loss)
                gc.collect()

        # Печать матрицы ошибок в консоль
        print(80 * "-")
        print("Общая матрица ошибок")
        print("Задача: {}".format("train"))
        print("Чекпоинт: {}".format(eparams["checkpoint_name"]))
        print(80 * "-")

        self.compute_metrics(total_y_gt, total_y_preds)
        gc.collect()

    def process_val_or_test(self, task):
        """
        Выполнение процесса тестирования или валидации модели:
        """
        # Списки для результатов (потери, точность,
        # истиные метки и метки предсказаний) выполнения тренировки за одну эпоху
        loss_testval = []
        acc_testval = []
        y_ground_truth = []
        y_predictions = []

        print("{} - задача в обработке".format(task))
        for data in tqdm(self.test_and_val_dataloader,):
            points, labels = data

            predictions, feature_transform = self.model(points)
            identity = torch.eye(feature_transform.shape[-1]).to(hparams['device'])
            reg_loss = torch.norm(
                identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1)))

            loss = F.nll_loss(predictions, labels.long()) + 0.001 * reg_loss
            loss_testval.append(loss.cpu().item())

            predictions = predictions.data.max(dim=1)[1]
            corrects = predictions.eq(labels.data).sum()
            accuracy = corrects.item() / predictions.numel()
            acc_testval.append(accuracy)

            labels = labels.view(-1, labels.numel()).squeeze(dim=0).tolist()
            predictions = predictions.view(-1, predictions.numel()).squeeze(dim=0).tolist()
            y_ground_truth.extend(labels)
            y_predictions.extend(predictions)

        print("\nПотеря: ", round(np.mean(loss_testval), 4))
        print("Точность:", round(np.mean(acc_testval), 4))

        return y_ground_truth, y_predictions

    def test_validate_model(self, task):
        y_gt, y_preds = self.process_val_or_test(task)
        # Print confusion matrix in console
        print(80 * "-")
        print("Матрица ошибок")
        print("Задача: {}".format(task))
        print(80 * "-")
        self.compute_metrics(y_gt, y_preds)
        gc.collect()

    def run_model(self, task="train"):
        # Get the proper dataloader for the proper task
        if task == "train":
            self.model = self.model.train()
            self.train_model()
        elif task == "validation":
            self.test_and_val_dataloader = self.data_loader[1]
            self.model = self.model.eval()
            self.test_validate_model(task)
        elif task == "test":
            self.test_and_val_dataloader = self.data_loader[2]
            self.model = self.model.eval()
            self.test_validate_model(task)

    @torch.no_grad()
    def watch_segmentation(self, dataset_test, random=False):
        """
        Визуализия результатов сегментации PointNet в одной комнате.

        Для визуализации взяты все точки одной комнаты, чтобы получить
        визуально сглаженное представление о комнате.

        Поскольку будут взяты все точки в комнатах, загрузчики данных
        не могут быть использованы, поскольку они возвращают меньшее количество
        точек на комнату/скользящее окно из-за процесса выборки.

        Есть два пути, как это сделать:
        1.- Читать напрямую из аннотированного файла (например, Area_6_office_33_annotated.txt)
        2.- Читать из файлов со скользящими окнами (например, Area_6_office_33_win14.pt)

        Последний вариант предпочтительнее, так как при желании можно также отображать информацию о каждом скользящем окне.

        Обзор рабочего процесса:
        1.- Выбрать случайным образом одно из доступных скользящих окон
        2.- Получить область N и пространство X из этого случайно выбранного скользящего окна
        3.- Получить все скользящие окна из Area_N Space_X
        4.- Получить информацию об объекте для каждого скользящего окна

        Выход:
        Dict, содержащий:
        - В качестве ключей: Идентификатор скользящего окна (WinID)
        - В качестве значений: Список списков. Каждый список содержит:
            - имя объекта (стул, стол,...)
            - Идентификатор объекта (из соответствующего dict)
            - количество обозначенных точек, которые есть у этого объекта в этой комнате (в виде тензора)
            - количество прогнозируемых точек для этого объекта в этой комнате (в виде тензора)
            - фактические прогнозируемые точки в относительных координатах (в виде тензора)
            - фактические прогнозируемые точки в абсолютных координатах (в виде тензора)
        """
        print("Визуализация сегментации данных")

        # Режим оценки работы модели
        self.model.eval()

        # Переменные для вычисления метрик
        per_win_y_true = []
        per_win_y_preds = []
        total_y_true = []
        total_y_preds = []

        # Выбрать объект для детекции
        # segmentation_target_object определен в settings.py
        # Из summary file доступны след dict:
        # 'all': {'ceiling': 0, 'clutter': 1, 'door': 2, 'floor': 3, 'wall': 4, 'beam': 5,
        #       'board': 6, 'bookcase': 7, 'chair': 8, 'table': 9, 'column': 10,
        #        'sofa': 11, 'window': 12, 'stairs': 13},
        # 'movable': {'clutter': 0, 'board': 1, 'bookcase': 2, 'chair': 3, 'table': 4, 'sofa': 5},
        # 'structural': {'ceiling': 0, 'clutter': 1, 'door': 2, 'floor': 3, 'wall': 4, 'beam': 5,
        #        'column': 6, 'window': 7, 'stairs': 8}}
        dict_to_use = self.objects_dict

        # Выбрать случайное скользящее окно в офисной комнате
        # (Например, 'Area_6_office_33_win14.pt')
        if random:
            picked_sliding_window = rnd.choice([i for i in dataset_test.sliding_windows if "office" in i])
            area_and_office = '_'.join(picked_sliding_window.split('_')[0:4])
        else:
            area_and_office = target_room_for_visualization

        print("Выбранный офис для визуализации: ", area_and_office)
        area_and_office += "_"

        # Получение всех скользящих окон, принадлежащих выбранной комнате
        # (н-р, 'Area_6_office_33_win0.pt', 'Area_6_office_33_win1.pt',...)
        all_sliding_windows_for_a_room = sorted([i for i in dataset_test.sliding_windows if area_and_office in i])

        # Загрузить тензоры из файлов скользящих окон
        room_tensors = []
        for f in all_sliding_windows_for_a_room:
            # Получить WinID
            winID = f.split('_')[-1].split('.')[0]

            path_to_sliding_window_file = os.path.join(
                path_to_current_sliding_windows_folder,
                f)
            room_tensors.append(
                (torch.load(path_to_sliding_window_file, map_location=torch.device(hparams["device"])),
                 winID
                 ))

        # Определим process bar, чтобы отображать процесс обработки файлов
        progress_bar = tqdm(room_tensors, total=len(room_tensors))

        # Определим выходной dict, содержащий точки для отображения в зависимости от WinID
        out_dict = dict()

        for (data_sl_wind, win_id) in progress_bar:
            msg = "{} - Splitting points".format(win_id)
            progress_bar.set_description(msg)

            # Количество столбцов, возвращаемых для каждой комнаты,
            # будет зависеть от того, должен ли учитываться цвет при вводе данных в модель
            # room -> [x_rel y_rel z_rel r g b x_abs y_abs x_abs winID label] (11 cols)
            points_rel = data_sl_wind[:, :hparams['dimensions_per_object']].to(hparams['device'])
            points_color = data_sl_wind[:, 3:6].to(hparams['device'])
            points_abs = data_sl_wind[:, 6:9].to(hparams['device'])
            target_labels = data_sl_wind[:, -1].to(hparams['device'])

            # По всем точкам в комнате определим, сколько из них принадлежит различным объектам
            # Мы собираемся поместить каждый объект в список списков, содержащий:
            # - object,
            # - идентификатор_объекта,
            # - количество аннотированных точек, которые этот объект имеет в этой комнате (в виде тензора)
            # - количество прогнозируемых точек для этого объекта в этой комнате (в виде тензора)
            #           (инициализируется равным нулю)
            # - предсказанные точки в относительных координатах (в виде тензора)
            # - предсказанные точки в абсолютных координатах (в виде тензора)
            point_breakdown = []
            for k, v in dict_to_use.items():
                point_breakdown.append([k, v, target_labels.eq(v).cpu().sum(), torch.tensor([0]), None, None])

            # Работаем с points_rel (вместо points_abs)
            # Сократим тензор данных, чтобы придать ему глубину batch_size = 1,
            # поскольку мы собираемся обрабатывать только одну комнату
            points = points_rel.unsqueeze(dim=0)

            # Тестирование модели
            # Model input: points.shape([batch_size, room_points, dimensons_per_point)]
            # Model output: preds.shape([batch_size, num_classes, room_points])
            # room_points - количество точек в комнате
            msg = "{} - Подача модели ({} точек)".format(win_id, len(points_rel))
            progress_bar.set_description(msg)
            preds, feature_transform = self.model(points)

            # Output после argmax: preds.shape([batch_size, room_points])
            preds = preds.data.max(1)[1]

            msg = "{} - Сохранение предсказаний".format(win_id)
            progress_bar.set_description(msg)

            # Сохранение прогнозов для каждого объекта
            for i in point_breakdown:
                # Выбор object_id элемента, чтобы проверить точность
                id = i[1]

                # Сохраняем прогнозы для этого объекта
                preds = torch.squeeze(preds, dim=0)
                i[3] = preds.eq(id).cpu().sum()

                # Использованные методы:
                # - torch.where() возвращает 1, если условие выполнено, 0 - иначе
                # - torch.nonzero() возвращает тензор, содержащий индексы всех ненулевых элементов ones_mask
                # - torch.index_select() возвращает новый тензор, который индексирует входной тензор вдоль
                ones_mask = torch.where(preds == id, 1., 0.).squeeze(dim=0)
                indices = torch.nonzero(ones_mask).squeeze(dim=1)

                # Сохраним точки для отображения
                # Точки для отображения - относительные координаты
                i[4] = torch.index_select(points_rel, 0, indices)
                # Точки для отображения - абсолютные координаты
                i[5] = torch.index_select(points_abs, 0, indices)

            # Сохранение результата
            out_dict[win_id] = point_breakdown

            # Подготовка данных для вычисления метрик
            targets = target_labels.view(-1, target_labels.numel()).squeeze(dim=0).tolist()
            preds = preds.view(-1, preds.numel()).squeeze(dim=0).tolist()
            per_win_y_true.append(targets)
            per_win_y_preds.append(preds)

        # Вычисление метрик
        print("Значения метрик для одного окна")
        for i in range(len(all_sliding_windows_for_a_room)):
            print("WinID: {} ".format(i) + 60 * "-")
            self.compute_metrics(per_win_y_true[i], per_win_y_preds[i])

        print(60 * "-")
        print("Значения метрик для всей комнаты {}:".format(area_and_office))
        print(60 * "-")
        # Создатеь единый список из всех списков
        for i in per_win_y_true:
            total_y_true.extend(i)

        for i in per_win_y_preds:
            total_y_preds.extend(i)

        self.compute_metrics(total_y_true, total_y_preds)

        # Visualize ground truth and resultant segmented points
        visualizer.render_segmentation(dict_to_use=dict_to_use,
                                       str_area_and_office=area_and_office,
                                       dict_model_segmented_points=out_dict,
                                       b_multiple_seg=True,
                                       b_hide_wall=True,
                                       draw_original_rgb_data=False,
                                       b_show_room_points=False)
