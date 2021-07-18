# import sys
#
# from hhutil.io import read_text
# from hanser.train.parser.parse import filter_log
#
#
# log_file = sys.argv[1]
# text = read_text(log_file)
# print(filter_log(text))

import sys
from hworkflow.project.base import merge_log_files

lines = merge_log_files([sys.argv[1]])
for l in lines:
    print(l)