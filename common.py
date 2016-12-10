from enum import Enum
from collections import namedtuple

LoadParams = namedtuple('LoadParams',
                        """batch_size load_size fine_size 
                        data_mean random_distort shuffle_window
                        collection_name
                        """)


class Mode(Enum):
    training = 1
    bn_callibration = 2
    testing = 3
