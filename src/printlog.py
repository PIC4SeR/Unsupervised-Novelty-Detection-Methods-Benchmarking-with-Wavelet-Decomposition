from rich import print as richprint
import logging, inspect, sys, os
import builtins

logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(message)s')


def print(*args, show=False, **kwargs):
    frame_info = inspect.stack()
    hierarchy = ''
    for frame in frame_info[1:-2]:
        hierarchy += f'{frame.filename.split("/")[-1]} at line {frame.lineno}'
        if frame != frame_info[-3]:
            hierarchy += ' -> '
    logging.info(hierarchy + ' - ' + ' '.join(map(str, args)))
    return richprint(*args, **kwargs) if show else None

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
