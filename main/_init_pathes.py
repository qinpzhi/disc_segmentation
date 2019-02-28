import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
add_path(osp.join(this_dir, '..'))

# Add data_loader to PYTHONPATH
# data_loader_path = osp.join(this_dir, '..', 'data_loader')
# add_path(data_loader_path)
# print data_loader_path

# Add util to PYTHONPATH
#utils_path = osp.join(this_dir, '..', 'utils')
#add_path(utils_path)
