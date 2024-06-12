# Import libraries
from summarizer import S3DIS_Summarizer
from dataset import *
from model import *
from trainer import *


def load_checkpoint(model):
    """
    Загрузка чекпоинта
    """
    # Путь к чекпоинту
    model_checkpoint = os.path.join(
        eparams['pc_data_path'],
        eparams['checkpoints_folder'],
        last_checkpoint
    )

    # Если чекпоинта нет - завершаем функцию
    if not os.path.exists(model_checkpoint):
        return

    # Загрузка чекпоинта
    print("Загрузка чекпоинта {} ...".format(model_checkpoint))
    state = torch.load(
        model_checkpoint,
        map_location=torch.device(hparams["device"]))
    model.load_state_dict(state['model'])

    return model


if __name__ == "__main__":
    task = ''.join(args.task)

    # предобработка датасета
    summarizer = S3DIS_Summarizer("models")
    objects_dict = summarizer.get_labels()

    summarizer.label_points_for_semantic_segmentation()
    summarizer.create_sliding_windows()

    # Создание датасетов
    dataset_train = S3DISDataset4SegmentationTrain('models', objects_dict, transform=None)
    dataset_val = S3DISDataset4SegmentationVal('models', objects_dict, transform=None)
    dataset_test = S3DISDataset4SegmentationTest('models', objects_dict, transform=None)

    dataset = dataset_train, dataset_val, dataset_test

    # создание dataloader'ов
    dataloaders = create_dataloaders(dataset)

    # Создание модели нейронной сети
    model = SegmentationPointNet(num_classes=hparams['num_classes'],
                                 point_dimension=hparams['dimensions_per_object'])
    tmp_model = load_checkpoint(model)
    # Если есть чекпоинт - загружаем его
    if tmp_model is not None:
        model = tmp_model

    trainer = Trainer(model, dataloaders, objects_dict)

    # Carry out the task to do
    if task != "watch":
        trainer.run_model(task)
    else:
        trainer.watch_segmentation(dataset_test, True)
