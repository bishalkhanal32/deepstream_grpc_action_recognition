from mmcv import load, dump
from pyskl.smp import *
data = load('aitdata.json')
tmpl = 'examples/extract_aitdatasets/AIT_dataset/{}'

lines = [(tmpl + ' {}').format(x['vid_name'], x['label']) for x in data]
mwlines(lines, 'aitdata.list')