
# import my libraries
from util import log


# import libraries
import pandas as pd



class Assign:

    def __init__(self, path):
        self.path = path
        self.col_map = []
        self.remap()

    def remap(self):
        df = pd.read_csv(self.path)
        self.col_map = [(col, 'cat') if df[col].dtype == object else
                        (col, 'quant') if df[col].dtype in [int, float] else
                        (col, '???') for col in df.columns]

    def remap_force(self, new_map):

        self.col_map = [(item[0], new_map.get(item[0])) if new_map.get(item[0], None) != None else
                        item for item in self.col_map]

    def log(self):
        log(self.path + "\n" + str(self.col_map), __name__, "info")


