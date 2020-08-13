import os
import shutil
from distutils.dir_util import copy_tree

from setuptools import find_packages, setup

# global variables
board = os.environ['BOARD']
repo_board_folder = f'Host_summernet'
board_notebooks_dir = os.environ['PYNQ_JUPYTER_NOTEBOOKS']
board_project_dir = os.path.join(board_notebooks_dir, 'summernet_deploy')

# check whether board is supported
def check_env():
    if not board == 'PYNQ-Z2':
        raise ValueError("Board {} is not supported.".format(board))
        
# check if the path already exists, delete if so
def check_path():
    if os.path.exists(board_project_dir):
        shutil.rmtree(board_project_dir)

# copy overlays to python package
def copy_overlays():
    src_ol_dir = os.path.join(repo_board_folder, 'overlay')
    dst_ol_dir = os.path.join(board_project_dir, 'overlay')
    copy_tree(src_ol_dir, dst_ol_dir)

# copy notebooks to jupyter home
def copy_notebooks():
    src_nb_dir = os.path.join(repo_board_folder)
    dst_nb_dir = os.path.join(board_project_dir)
    copy_tree(src_nb_dir, dst_nb_dir)
    
check_env()
check_path()
copy_overlays()
copy_notebooks()

setup(
    name="Hardware-Acceleration-Circuit-Design-of-Object-Detection-Network-Based-on-FPGA",
    version='1.0',
    install_requires=[
        'pynq>=2.3',
    ],
    url='https://github.com/ZhangYuQAQ/Hardware-Acceleration-Circuit-Design-of-Object-Detection-Network-Based-on-FPGA',
    license='MIT License',
    author="zhangyu",
    author_email="1433213806@qq.com",
    packages=find_packages(),
    description="PYNQ example of a Hardware-Acceleration.")