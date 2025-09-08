import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class LabelsForDxData(object):

    def __init__(self, ccs_multi_dx_file):
        self.ccs_multi_dx_df = pd.read_csv(ccs_multi_dx_file, header=0, dtype=object)
        self.code2first_level_dx = {}  # label clustering data
        self.build_maps()

    def build_maps(self):
        for i, row in self.ccs_multi_dx_df.iterrows():
            code = row[0][1:-1].strip()
            level_1_cat = row[1][1:-1].strip()
            self.code2first_level_dx[code] = level_1_cat
        print('len(code2first_level_dx):', len(self.code2first_level_dx))


class LabelsForPrData(object):

    def __init__(self, ccs_multi_pr_file):
        self.ccs_multi_pr_df = pd.read_csv(ccs_multi_pr_file, header=0, dtype=object)
        self.code2first_level_pr = {}  # label clustering data
        self.build_maps()

    def build_maps(self):
        for i, row in self.ccs_multi_pr_df.iterrows():
            code = row[0][1:-1].strip()
            level_1_cat = row[1][1:-1].strip()
            self.code2first_level_pr[code] = level_1_cat
        print('len(code2first_level_pr):', len(self.code2first_level_pr))


class LabelsForData(object):

    def __init__(self, ccs_multi_dx_file, ccs_single_dx_file):
        self.ccs_multi_dx_df = pd.read_csv(ccs_multi_dx_file, header=0, dtype=object)
        self.ccs_single_dx_df = pd.read_csv(ccs_single_dx_file, header=0, dtype=object)
        self.code2single_dx = {}  # label sequential diagnosis prediction data
        self.code2first_level_dx = {}  # label clustering data
        self.build_maps()

    def build_maps(self):
        for i, row in self.ccs_multi_dx_df.iterrows():
            code = row[0][1:-1].strip()
            level_1_cat = row[1][1:-1].strip()
            self.code2first_level_dx[code] = level_1_cat

        for i, row in self.ccs_single_dx_df.iterrows():
            code = row[0][1:-1].strip()
            single_cat = row[1][1:-1].strip()
            self.code2single_dx[code] = single_cat
        print('len(code2first_level_dx):', len(self.code2first_level_dx))
        print('len(code2single_dx):', len(self.code2single_dx))