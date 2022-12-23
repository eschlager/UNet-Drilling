# -*- coding: utf-8 -*-
"""
Created on 27.10.2020
@author: eschlager
"""

import logging


def define_root_logger(filename='logfile.txt'):
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(filename)-18s - %(message)s',
                        filename=filename,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)
