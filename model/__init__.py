from .lenet300 import LeNet300, lenet300_classic, lenet300_classic_drop, lenet300_modern, lenet300_modern_drop
from .lenet5 import LeNet5, lenet5_classic, lenet5_drop
from .nincif import nincif_bn
from .resnet_all import resnet56, resnet110
from .vggcif import vggcif9_bn, vggcif11_bn, vggcif13_bn, vggcif16_bn, vggcif19_bn
from .densenet_all import densenet40
from .googlenet_all import googlenet
__all__ = ['lenet300_classic',
           'lenet5_classic', 'lenet5_drop', 'resnet56', 'resnet110',
           'vggcif16_bn', 'vggcif19_bn',
           'densenet40',
           'googlenet']