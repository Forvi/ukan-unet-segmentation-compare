from geoseg.models.UNetFormer import UNetFormer


class UNetFormerModel(UNetFormer):
    """
    TODO: дописать докстринг
    
    Модель UNetFormer. Класс наследует объект UNetFormer из репозитория GeoSeg.

    Args:
        decode_channels 
        dropout
        backbone_name'
        pretrained
        window_size
        num_classes

    Attributes:
    
    Example:

    """
    def __init__(self, 
                 decode_channels=64, 
                 dropout=0.1, 
                 backbone_name='swsl_resnet18', 
                 pretrained=True, 
                 window_size=8, 
                 num_classes=6):
        super().__init__(decode_channels, dropout, backbone_name, pretrained, window_size, num_classes)