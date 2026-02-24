import glog as logger
import sys
import os
import torch
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def set_log_file(fname, file_only=False):
    """
    Cross-platform replacement for Linux `tee`.
    Captures:
    - print()
    - logger output
    - Python tracebacks
    Works on Windows, Linux, SSH, everything.
    """
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    log_file = open(fname, 'w', buffering=1)

    class TeeStream:
        def __init__(self, terminal, logfile, file_only=False):
            self.terminal = terminal
            self.logfile = logfile
            self.file_only = file_only

        def write(self, message):
            self.logfile.write(message)
            if not self.file_only:
                self.terminal.write(message)

        def flush(self):
            self.logfile.flush()
            if not self.file_only:
                self.terminal.flush()

    # Replace stdout and stderr
    sys.stdout = TeeStream(sys.__stdout__, log_file, file_only)
    sys.stderr = TeeStream(sys.__stderr__, log_file, file_only)

    # Make glog write to the same stream
    logger.logger.handlers[0].stream = sys.stdout
    logger.handler.stream = sys.stdout


def print_args(args):
    logger.info('-------- args -----------')
    for k, v in vars(args).items():
        logger.info('%s: ' % k + str(v))
    logger.info('-------------------------')


def str_in_list(x, str_list):
    for s in str_list:
        if s in x:
            return True
    return False


def myitem(x):
    if torch.is_tensor(x):
        return x.item()
    return x
