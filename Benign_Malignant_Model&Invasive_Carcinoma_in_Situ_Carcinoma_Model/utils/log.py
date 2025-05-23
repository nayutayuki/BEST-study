'''
@Date: 2019-12-05 04:23:25
@Author: Yong Pi
@LastEditors: Yong Pi
@LastEditTime: 2019-12-05 04:23:26
@Description: All rights reserved.
'''
import logging


class Logger(object):
    def __init__(self, filename, level=logging.INFO,
                 format='%(asctime)s %(levelname)s %(message)s',
                 datefmt='%a, %d %b %Y %H:%M:%S', filemode='w'):
        self.level = level
        self.format = format
        self.datefmt = datefmt
        self.filename = filename
        self.filemode = filemode
        logging.basicConfig(level=self.level,
                            format=self.format,
                            datefmt=self.datefmt,
                            filename=self.filename,
                            filemode=self.filemode)
        self._set_streaming_handler()

    def _set_streaming_handler(self, level=logging.INFO, formatter='%(asctime)s %(levelname)-8s %(message)s'):
        console = logging.StreamHandler()
        console.setLevel(level)
        curr_formatter = logging.Formatter(formatter)
        console.setFormatter(curr_formatter)
        logging.getLogger(self.filename).addHandler(console)

    def get_logger(self):
        return logging.getLogger(self.filename)


def get_logger(log_path):
    logging.basicConfig(level=logging.INFO,
                        filename=log_path,
                        format='%(levelname)s:%(name)s:%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.level = logging.NOTSET
    logger = logging.getLogger()
    logger.addHandler(console)
    return logger
