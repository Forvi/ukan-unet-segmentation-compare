from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import Cityscapes, VOCDetection
import torchvision.transforms as transforms

class CityscapesDataset(Dataset):
    """
    Класс для работы с датасетом Cityscapes.

    Attributes
    ----------
    root : str
        Путь до директории с данными
    split : str
        Тип выборки данных для тренировки и тестирования
        Может быть: train, test, val.
    mode : str
        Режим для качества датасета
        Может быть: fine, corse
        В проекте используется fine
    target_type : str
        Тип целевой переменной в использовании
        Может быть: instance, semantic, polygon or color
    transform : callable, optional
        Функция/преобразование, которое принимает изображение PIL и возвращает преобразованную версию
    target_transform : callable, optional
        Функция/преобразование, которое принимает цель и преобразует ее.

    Methods
    -------
    __len__():
        Возвращает размер датасета
    __getitem__(index):
        Возвращает элементы image и target по заданному индексу
    get_loaders(root, batch_size, transform, target_transform)
        Возвращает выборки для обучения и тестирования

    Example
    -------
        path = './data/Cityscapes'
        dataset = CityscapesDataset(root=path, split='train')
        print(dataset.__getitem__(2))
    """

    def __init__(self, root: str, split: str='train', mode: str='fine', 
                 target_type: str ='semantic', transform=None, target_transform=None):
        self.dataset = Cityscapes(
            root=root,
            split=split,
            mode=mode,
            target_type=target_type,
            transform=transform,
            target_transform=target_transform)


    def __len__(self):
        """Возвращает размер датасета"""
        return len(self.dataset)


    def __getitem__(self, index):
        """Возвращает элементы image и target по заданному индексу"""
        img, smnt = self.dataset[index]
        return img, smnt
    

    @staticmethod
    def get_loaders(root: str, batch_size: int=32, transform=None, target_transform=None, num_workers: int=0):
        """
        Возвращает выборки для обучения и тестирования.

        Parameters
        ----------
        root : str
            Путь до директории с данными
        batch_size : int
            Размер батча
        transform : callable, optional
            Функция/преобразование, которое принимает изображение PIL и возвращает преобразованную версию
        target_transform : callable, optional
            Функция/преобразование, которое принимает цель и преобразует ее.
        num_workers : int
            Количество подпроцессов использованных для загрузки данных
        Returns 
        -------
            Тренировчную и тестовую выборки train_loader, val_loader
        """
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256, 512)),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        if target_transform is None:
            target_transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST),
            ])

        train_dataset = CityscapesDataset(root=root, split='train', transform=transform, target_transform=target_transform)
        val_dataset = CityscapesDataset(root=root, split='val', transform=transform, target_transform=target_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader