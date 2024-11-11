import os

import get_cosmopower_emus
import class_sz_data

def get_cosmopower_path():
    get_cosmopower_emus.set_path()
    return os.getenv('PATH_TO_CLASS_SZ_DATA')

path_to_class_sz_data = get_cosmopower_path()
class_sz_data.get_data_from_class_sz_repo(path_to_class_sz_data)