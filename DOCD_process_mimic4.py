import pickle
import csv
import os
import sys
import numpy as np
import sklearn.model_selection as ms
import torch
import time
from torch.utils.data import TensorDataset
from tqdm import tqdm
import random
from build_trees import build_trees_diag, build_trees_proc
from data_labelling import LabelsForDxData, LabelsForPrData
import pandas as pd


class EncounterInfo:
    def __init__(self, patient_id, encounter_id, expired):
        self.patient_id = patient_id
        self.encounter_id = encounter_id
        self.expired = expired
        self.demographics = []
        self.vital_signs = []
        self.dx_ids = []
        self.dx_ccs_cat1 = []
        self.proc_ids = []
        self.proc_ccs_cat1 = []


class EncounterFeatures:
    def __init__(self,
                 patient_id,
                 label_expired,
                 label_dx_ccs_cat1,
                 label_proc_ccs_cat1,
                 demographics_ints,
                 vital_signs_ints,
                 dx_ids,
                 dx_ints,
                 proc_ids,
                 proc_ints):
        self.patient_id = patient_id
        self.label_expired = label_expired
        self.label_dx_ccs_cat1 = label_dx_ccs_cat1
        self.label_proc_ccs_cat1 = label_proc_ccs_cat1
        self.demographics_ints = demographics_ints
        self.vital_signs_ints = vital_signs_ints
        self.dx_ids = dx_ids
        self.dx_ints = dx_ints
        self.proc_ids = proc_ids
        self.proc_ints = proc_ints
        self.dx_mask = None
        self.proc_mask = None
        self.prior_indices = None
        self.prior_values = None


def process_hospital_detail(infile, encounter_dict):
    patients = 0
    lines = 0
    with open(infile, 'r') as f:
        patient_dict = {}
        for line in csv.DictReader(f):
            lines += 1
            patient_id = line['subject_id']
            if patient_id not in patient_dict:
                patient_dict[patient_id] = []
                patients += 1
    print('number of patients: {}'.format(patients))

    with open(infile, 'r') as f:
        count = 0
        death_num = 0
        alive_num = 0
        for line in tqdm(csv.DictReader(f)):
            patient_id = line['subject_id']
            encounter_id = line['hadm_id']
            discharge_status = int(line['hospital_expire_flag'])
            expired = True if discharge_status == 1 else False
            if expired:
                death_num += 1
            else:
                alive_num += 1
            ei = EncounterInfo(patient_id, encounter_id, expired)
            encounter_dict[encounter_id] = ei
            count += 1
    print('Number of expired records: ', death_num)
    print('Number of survived records: ', alive_num)
    return encounter_dict


def processs_demographics(infile, encounter_dict):
    demographic_columns = ['Gender', 'Age', 'Race', 'Admission_type', 'Admission_location', 'Discharge_location',
                           'Marital_status', 'Insurance', 'Height']
    with open(infile, 'r') as f:
        for line in tqdm(csv.DictReader(f)):
            encounter_id = line['hadm_id']
            demographics_data = [line[col] if line[col] else '-1' for col in demographic_columns]
            encounter_dict[encounter_id].demographics = demographics_data
    return encounter_dict


def processs_vital_signs(infile, encounter_dict):
    with open(infile, 'r') as f:
        for line in tqdm(csv.DictReader(f)):
            encounter_id = line['hadm_id']
            vital_signs_data = [value if value else '-1' for key, value in line.items() if key != 'hadm_id']
            encounter_dict[encounter_id].vital_signs = vital_signs_data
    return encounter_dict


def process_HospDiagnosis(infile, icd10toicd9_file, multi_dx_file, encounter_dict):
    icd10cmtoicd9gem_df = pd.read_csv(icd10toicd9_file, header=0, sep=',', quotechar='"')
    icd10cmtoicd9 = {}
    for index, row in icd10cmtoicd9gem_df.iterrows():
        icd10cmtoicd9[row.icd10cm] = row.icd9cm
    label4data = LabelsForDxData(multi_dx_file)
    with open(infile, 'r') as f:
        count = 0
        missing = 0
        NoMapping = 0
        for line in tqdm(csv.DictReader(f)):
            encounter_id = line['hadm_id']
            if encounter_id == '24146561':
                continue
            if encounter_id == '29119250':
                continue
            dx = line['icd_code'].strip()
            if len(dx) == 0:
                missing += 1
                continue
            icd_version = line['icd_version'].strip()
            if icd_version == '10':
                if dx in icd10cmtoicd9:
                    dx = icd10cmtoicd9[dx]
                    if dx == 'NoDx':
                        NoMapping += 1
                        continue
                else:
                    NoMapping += 1
                    continue
            dx_id = 'D_' + dx
            dx_ccs_cat1 = 'D_' + label4data.code2first_level_dx[dx]
            if dx_id not in encounter_dict[encounter_id].dx_ids:
                encounter_dict[encounter_id].dx_ids.append(dx_id)
                encounter_dict[encounter_id].dx_ccs_cat1.append(dx_ccs_cat1)
            count += 1
    print('hopsital diagnosis with encounter_id: {}'.format(count))
    print('encounter_id without diagnosis:', missing)
    print('diagnosis 10to9Mapping don't exist:', NoMapping)
    return encounter_dict


def process_HospProcedures(infile, icd10toicd9_file, multi_proc_file, encounter_dict):
    icd10pcstoicd9gem_df = pd.read_csv(icd10toicd9_file, header=0, sep=',', quotechar='"')
    icd10pcstoicd9 = {}
    proc_failed = []
    for index, row in icd10pcstoicd9gem_df.iterrows():
        icd10pcstoicd9[row.icd10cm] = row.icd9cm
    label4data = LabelsForPrData(multi_proc_file)
    with open(infile, 'r') as f:
        missing = 0
        count = 0
        NoMapping = 0
        for line in tqdm(csv.DictReader(f)):
            encounter_id = line['hadm_id']
            if encounter_id == '24146561':
                continue
            proc = str(line['icd_code'].strip())
            if len(proc) == 0:
                missing += 1
                continue
            icd_version = line['icd_version'].strip()
            if icd_version == '10':
                if proc in icd10pcstoicd9:
                    proc = str(icd10pcstoicd9[proc])
                else:
                    NoMapping += 1
                    continue
            for _ in range(4):
                if proc in label4data.code2first_level_pr:
                    proc_id = 'P_' + proc
                    proc_ccs_cat1 = 'P_' + label4data.code2first_level_pr[proc]
                    break
                else:
                    proc = '0' + proc
            else:
                proc_failed.append(proc)
            if proc_id not in encounter_dict[encounter_id].proc_ids:
                encounter_dict[encounter_id].proc_ids.append(proc_id)
                encounter_dict[encounter_id].proc_ccs_cat1.append(proc_ccs_cat1)
            count += 1
    print('hopsital procedures with encounter_id: {}'.format(count))
    print('encounter_id without procedures:', missing)
    print('procedures 10to9Mapping don't exist:', NoMapping)
    return encounter_dict


def get_encounter_features(encounter_dict, skip_duplicate=False, min_num_codes=1, max_num_codes=30):
    key_list = []
    key_list_live = []
    key_list_death = []
    enc_features_list = []
    demographics_str2int = {}
    vital_signs_str2int = {}
    dx_str2int = {}
    dx_ccs_cat1_str2int = {}
    proc_str2int = {}
    proc_ccs_cat1_str2int = {}
    diag_vocab_set = {}
    proc_vocab_set = {}
    num_cut = 0
    num_duplicate = 0
    count = 0
    num_dx_ids = 0
    num_proc_ids = 0
    num_unique_dx_ids = 0
    num_unique_proc_ids = 0
    min_dx_cut = 0
    min_proc_cut = 0
    max_dx_cut = 0
    max_proc_cut = 0

    directory = 'processed_data/mimiciv/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    pickle.dump(encounter_dict, open(os.path.join(directory, 'encounter_dict.p'), 'wb'))

    for enc in encounter_dict.values():
        for demographics in enc.demographics:
            if demographics not in demographics_str2int:
                demographics_str2int[demographics] = len(demographics_str2int)
        for vital_signs in enc.vital_signs:
            if vital_signs not in vital_signs_str2int:
                vital_signs_str2int[vital_signs] = len(vital_signs_str2int)
        for dx_ccs_cat1 in enc.dx_ccs_cat1:
            if dx_ccs_cat1 not in dx_ccs_cat1_str2int:
                dx_ccs_cat1_str2int[dx_ccs_cat1] = len(dx_ccs_cat1_str2int)
        for proc_ccs_cat1 in enc.proc_ccs_cat1:
            if proc_ccs_cat1 not in proc_ccs_cat1_str2int:
                proc_ccs_cat1_str2int[proc_ccs_cat1] = len(proc_ccs_cat1_str2int)

        for proc_id in enc.proc_ids:
            if proc_id in proc_vocab_set:
                proc_vocab_set[proc_id] += 1
            else:
                proc_vocab_set[proc_id] = 1
        sorted_vocab = {k: v for k, v in sorted(proc_vocab_set.items(), key=lambda item: item[1], reverse=True)}
        outfd = open(os.path.join(directory, 'proc_vocab.txt'), 'w')
        for k, v in sorted_vocab.items():
            outfd.write(k + '\n')
        outfd.close()

        for dx_id in enc.dx_ids:
            if dx_id in diag_vocab_set:
                diag_vocab_set[dx_id] += 1
            else:
                diag_vocab_set[dx_id] = 1
        sorted_vocab = {k: v for k, v in sorted(diag_vocab_set.items(), key=lambda item: item[1], reverse=True)}
        outfd = open(os.path.join(directory, 'diag_vocab.txt'), 'w')
        for k, v in sorted_vocab.items():
            outfd.write(k + '\n')
        outfd.close()
    dx_str2int = build_trees_diag('processed_data/mimiciv/diag_vocab.txt', 'ccs/ccs_multi_dx_tool_2015.csv', 'processed_data/mimiciv/')
    print('finish diagnosis tree building')
    proc_str2int = build_trees_proc('processed_data/mimiciv/proc_vocab.txt', 'ccs/ccs_multi_pr_tool_2015.csv', 'processed_data/mimiciv/')
    print('finish procedure tree building')
    pickle.dump(dx_str2int, open(os.path.join(directory, 'dx_map.p'), 'wb'))
    pickle.dump(proc_str2int, open(os.path.join(directory, 'proc_map.p'), 'wb'))
    pickle.dump(dx_ccs_cat1_str2int, open(os.path.join(directory, 'dx_ccs_cat1_map.p'), 'wb'))
    pickle.dump(proc_ccs_cat1_str2int, open(os.path.join(directory, 'proc_ccs_cat1_map.p'), 'wb'))

    for _, enc in encounter_dict.items():
        if skip_duplicate:
            if (len(enc.dx_ids) > len(set(enc.dx_ids)) or len(enc.proc_ids) > len(set(enc.proc_ids))):
                num_duplicate += 1
                continue
        if len(enc.dx_ids) < min_num_codes:
            min_dx_cut += 1
            continue
        if len(enc.proc_ids) < min_num_codes:
            min_proc_cut += 1
            continue
        if len(enc.dx_ids) > max_num_codes:
            max_dx_cut += 1
            continue
        if len(enc.proc_ids) > max_num_codes:
            max_proc_cut += 1
            continue
        count += 1

        num_dx_ids += len(enc.dx_ids)
        num_proc_ids += len(enc.proc_ids)
        num_unique_dx_ids += len(set(enc.dx_ids))
        num_unique_proc_ids += len(set(enc.proc_ids))

        patient_id = enc.patient_id + ':' + enc.encounter_id
        if enc.expired:
            label_expired = 1
        else:
            label_expired = 0
        if label_expired == 1:
            key_list_death.append(patient_id)
        else:
            key_list_live.append(patient_id)

        dx_ints = [dx_str2int[item] for item in enc.dx_ids]
        proc_ints = [proc_str2int[item] for item in enc.proc_ids]
        demographics_ints = [demographics_str2int[item] for item in enc.demographics]
        vital_signs_ints = [vital_signs_str2int[item] for item in enc.vital_signs]

        dx_ccs_cat1 = [dx_ccs_cat1_str2int[item] for item in enc.dx_ccs_cat1]
        dx_ccs_cat1_onehot = np.zeros(len(dx_ccs_cat1_str2int), dtype=np.int32)
        dx_ccs_cat1_onehot[dx_ccs_cat1] = 1.0

        proc_ccs_cat1 = [proc_ccs_cat1_str2int[item] for item in enc.proc_ccs_cat1]
        proc_ccs_cat1_onehot = np.zeros(len(proc_ccs_cat1_str2int), dtype=np.int32)
        proc_ccs_cat1_onehot[proc_ccs_cat1] = 1.0

        enc_features = EncounterFeatures(patient_id, label_expired, dx_ccs_cat1_onehot, proc_ccs_cat1_onehot, demographics_ints, vital_signs_ints, enc.dx_ids, dx_ints, enc.proc_ids, proc_ints)
        key_list.append(patient_id)
        enc_features_list.append(enc_features)

    for ef in enc_features_list:
        dx_padding_idx = len(dx_str2int)
        proc_padding_idx = len(proc_str2int)
        if len(ef.dx_ints) < max_num_codes:
            ef.dx_ints.extend([dx_padding_idx] * (max_num_codes - len(ef.dx_ints)))
        if len(ef.proc_ints) < max_num_codes:
            ef.proc_ints.extend([proc_padding_idx] * (max_num_codes - len(ef.proc_ints)))
        ef.dx_mask = [0 if i == dx_padding_idx else 1 for i in ef.dx_ints]
        ef.proc_mask = [0 if i == proc_padding_idx else 1 for i in ef.proc_ints]

    pickle.dump(enc_features_list, open(os.path.join(directory, 'enc_features_list.p'), 'wb'))
    pickle.dump(key_list, open(os.path.join(directory, 'key_list.p'), 'wb'))
    pickle.dump(key_list_live, open(os.path.join(directory, 'key_list_live.p'), 'wb'))
    pickle.dump(key_list_death, open(os.path.join(directory, 'key_list_death.p'), 'wb'))

    print('Filtered encounters due to duplicate codes: %d' % num_duplicate)
    print('Filtered encounters due to thresholding: %d' % num_cut)
    print('Average num_dx_ids: %f' % (num_dx_ids / count))
    print('Average num_proc_ids: %f' % (num_proc_ids / count))
    print('Average num_unique_dx_ids: %f' % (num_unique_dx_ids / count))
    print('Average num_unique_proc_ids: %f' % (num_unique_proc_ids / count))
    print('Min dx cut: %d' % min_dx_cut)
    print('Max dx cut: %d' % max_dx_cut)
    print('Min proc cut: %d' % min_proc_cut)
    print('Max proc cut: %d' % max_proc_cut)

    print('Number of included records: %d' % len(key_list))
    print('Number of included expired records: %d' % len(key_list_death))
    print('Number of included alive records: %d' % len(key_list_live))
    print('Vocab sizes - demographics_ints:', len(demographics_str2int), 'vital_signs_ints:', len(vital_signs_str2int), 'dx_ints:', len(dx_str2int), 'proc_ints:', len(proc_str2int), 'dx_ccs_cat1:', len(dx_ccs_cat1_str2int), 'proc_ccs_cat1:', len(proc_ccs_cat1_str2int))

    return key_list, enc_features_list, key_list_live, key_list_death


def select_train_valid_test(key_list, random_seed=1234):
    key_train, key_temp = ms.train_test_split(key_list, test_size=0.2, random_state=random_seed)
    key_valid, key_test = ms.train_test_split(key_temp, test_size=0.5, random_state=random_seed)
    return key_train, key_valid, key_test


def count_conditional_prob_dp(enc_features_list, output_path, train_key_set=None):
    dx_freqs = {}
    proc_freqs = {}
    dp_freqs = {}
    total_visit = 0
    for enc_feature in enc_features_list:
        key = enc_feature.patient_id
        if (train_key_set is not None and key not in train_key_set):
            total_visit += 1
            continue
        dx_ids = enc_feature.dx_ids
        proc_ids = enc_feature.proc_ids
        for dx in dx_ids:
            if dx not in dx_freqs:
                dx_freqs[dx] = 0
            dx_freqs[dx] += 1
        for proc in proc_ids:
            if proc not in proc_freqs:
                proc_freqs[proc] = 0
            proc_freqs[proc] += 1
        for dx in dx_ids:
            for proc in proc_ids:
                dp = dx + ',' + proc
                if dp not in dp_freqs:
                    dp_freqs[dp] = 0
                dp_freqs[dp] += 1
        total_visit += 1

    dx_probs = dict([(k, v / float(total_visit)) for k, v in dx_freqs.items()])
    proc_probs = dict([(k, v / float(total_visit)) for k, v in proc_freqs.items()])
    dp_probs = dict([(k, v / float(total_visit)) for k, v in dp_freqs.items()])

    dp_cond_probs = {}
    pd_cond_probs = {}
    for dx, dx_prob in dx_probs.items():
        for proc, proc_prob in proc_probs.items():
            dp = dx + ',' + proc
            pd = proc + ',' + dx
            if dp in dp_probs:
                dp_cond_probs[dp] = dp_probs[dp] / dx_prob
                pd_cond_probs[pd] = dp_probs[dp] / proc_prob
            else:
                dp_cond_probs[dp] = 0.0
                pd_cond_probs[pd] = 0.0
    pickle.dump(dp_cond_probs, open(os.path.join(output_path, 'dp_cond_probs.empirical.p'), 'wb'))
    pickle.dump(pd_cond_probs, open(os.path.join(output_path, 'pd_cond_probs.empirical.p'), 'wb'))


def add_sparse_prior_guide_dp(enc_features_list, stats_path, key_set=None, max_num_codes=30):
    dp_cond_probs = pickle.load(open(os.path.join(stats_path, 'dp_cond_probs.empirical.p'), 'rb'))
    pd_cond_probs = pickle.load(open(os.path.join(stats_path, 'pd_cond_probs.empirical.p'), 'rb'))

    print('Adding prior guide')
    total_visit = 0
    new_enc_features_list = []
    for enc_features in enc_features_list:
        key = enc_features.patient_id
        if (key_set is not None and key not in key_set):
            total_visit += 1
            continue
        dx_ids = enc_features.dx_ids
        proc_ids = enc_features.proc_ids
        indices = []
        values = []
        for i, dx in enumerate(dx_ids):
            for j, proc in enumerate(proc_ids):
                dp = dx + ',' + proc
                indices.append((i, max_num_codes + j))
                prob = 0.0 if dp not in dp_cond_probs else dp_cond_probs[dp]
                values.append(prob)
        for i, proc in enumerate(proc_ids):
            for j, dx in enumerate(dx_ids):
                pd = proc + ',' + dx
                indices.append((max_num_codes + i, j))
                prob = 0.0 if pd not in pd_cond_probs else pd_cond_probs[pd]
                values.append(prob)

        enc_features.prior_indices = indices
        enc_features.prior_values = values
        new_enc_features_list.append(enc_features)

        total_visit += 1
    return new_enc_features_list


def count_conditional_prob_dp_orderset_bylabel(enc_features_list, output_path, train_key_set=None, label="expired"):
    dx_freqs = {}
    proc_freqs = {}
    dp_freqs = {}
    total_visit = 0
    for enc_feature in enc_features_list:
        key = enc_feature.patient_id
        if (train_key_set is not None and key not in train_key_set):
            total_visit += 1
            continue
        dx_ids = enc_feature.dx_ids
        proc_ids = enc_feature.proc_ids
        for dx in dx_ids:
            if dx not in dx_freqs:
                dx_freqs[dx] = 0
            dx_freqs[dx] += 1
        for proc in proc_ids:
            if proc not in proc_freqs:
                proc_freqs[proc] = 0
            proc_freqs[proc] += 1
        for dx in dx_ids:
            for proc in proc_ids:
                dp = dx + ',' + proc
                if dp not in dp_freqs:
                    dp_freqs[dp] = 0
                dp_freqs[dp] += 1
        total_visit += 1

    dx_probs = dict([(k, v / float(total_visit)) for k, v in dx_freqs.items()])
    proc_probs = dict([(k, v / float(total_visit)) for k, v in proc_freqs.items()])
    dp_probs = dict([(k, v / float(total_visit)) for k, v in dp_freqs.items()])

    dp_cond_probs = {}
    pd_cond_probs = {}
    for dx, dx_prob in dx_probs.items():
        for proc, proc_prob in proc_probs.items():
            dp = dx + ',' + proc
            pd = proc + ',' + dx
            if dp in dp_probs:
                dp_cond_probs[dp] = dp_probs[dp] / dx_prob
                pd_cond_probs[pd] = dp_probs[dp] / proc_prob
            else:
                dp_cond_probs[dp] = 0.0
                pd_cond_probs[pd] = 0.0
    if label == "expired":
        pickle.dump(dp_cond_probs, open(os.path.join(output_path, 'dp_cond_probs_orderset_expired.empirical.p'), 'wb'))
        pickle.dump(pd_cond_probs, open(os.path.join(output_path, 'pd_cond_probs_orderset_expired.empirical.p'), 'wb'))


def add_sparse_prior_guide_dp_orderset_bylabel(enc_features_list,
                                               stats_path,
                                               key_set=None,
                                               max_num_codes=30,
                                               label="expired"):
    if label == "expired":
        dp_cond_probs = pickle.load(open(os.path.join(stats_path, 'dp_cond_probs_orderset_expired.empirical.p'), 'rb'))
        pd_cond_probs = pickle.load(open(os.path.join(stats_path, 'pd_cond_probs_orderset_expired.empirical.p'), 'rb'))
    print('Adding prior guide')
    new_enc_features_list = []
    all_samples_dict = {}
    for enc_features in enc_features_list:
        sampleid = enc_features.patient_id
        all_samples_dict[sampleid] = enc_features
    for key in key_set:
        enc_features = all_samples_dict[key]
        dx_ids = enc_features.dx_ids
        proc_ids = enc_features.proc_ids

        indices = []
        values = []
        for i, dx in enumerate(dx_ids):
            for j, proc in enumerate(proc_ids):
                dp = dx + ',' + proc
                indices.append((i, max_num_codes + j))
                prob = 0.0 if dp not in dp_cond_probs else dp_cond_probs[dp]
                values.append(prob)
        for i, proc in enumerate(proc_ids):
            for j, dx in enumerate(dx_ids):
                pd = proc + ',' + dx
                indices.append((max_num_codes + i, j))
                prob = 0.0 if pd not in pd_cond_probs else pd_cond_probs[pd]
                values.append(prob)

        enc_features.prior_indices = indices
        enc_features.prior_values = values
        new_enc_features_list.append(enc_features)

    return new_enc_features_list


def convert_features_to_tensors(enc_features):
    all_expired_labels = torch.tensor([f.label_expired for f in enc_features], dtype=torch.long)
    all_label_dx_ccs_cat1 = torch.tensor([f.label_dx_ccs_cat1 for f in enc_features], dtype=torch.long)
    all_label_proc_ccs_cat1 = torch.tensor([f.label_proc_ccs_cat1 for f in enc_features], dtype=torch.long)
    all_demographics_ints = torch.tensor([f.demographics_ints for f in enc_features], dtype=torch.long)
    all_vital_signs_ints = torch.tensor([f.vital_signs_ints for f in enc_features], dtype=torch.long)
    all_dx_ints = torch.tensor([f.dx_ints for f in enc_features], dtype=torch.long)
    all_dx_masks = torch.tensor([f.dx_mask for f in enc_features], dtype=torch.float)
    all_proc_ints = torch.tensor([f.proc_ints for f in enc_features], dtype=torch.long)
    all_proc_masks = torch.tensor([f.proc_mask for f in enc_features], dtype=torch.float)
    dataset = TensorDataset(all_dx_ints, all_dx_masks, all_proc_ints, all_proc_masks, all_demographics_ints, all_vital_signs_ints, all_expired_labels, all_label_dx_ccs_cat1, all_label_proc_ccs_cat1)
    return dataset


def get_prior_guide(enc_features):
    prior_guide_list = []
    for feats in enc_features:
        indices = torch.tensor(list(zip(*feats.prior_indices))).reshape(2, -1)
        values = torch.tensor(feats.prior_values)
        prior_guide_list.append((indices, values))
    return prior_guide_list


def get_mimiciv_datasets(data_dir, fold=0, flag="expired"):
    hospital_detail_file = os.path.join(data_dir, 'encodedVisit.csv')
    HospDiagnosis_file = os.path.join(data_dir, 'HospDiagnosis.csv')
    HospProcedures_file = os.path.join(data_dir, 'HospProcedures.csv')
    vital_signs_file = os.path.join(data_dir, 'eventStatistics.csv')
    multi_dx_file = os.path.join('ccs', 'ccs_multi_dx_tool_2015.csv')
    icd10toicd9dx_file = os.path.join('ccs', 'icd10cmtoicd9gem.csv')
    multi_proc_file = os.path.join('ccs', 'ccs_multi_pr_tool_2015.csv')
    icd10toicd9proc_file = os.path.join('ccs', 'icd10pcstoicd9gem.csv')

    fold_path = os.path.join(data_dir, 'fold_{}'.format(fold))
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
    stats_path = os.path.join(fold_path, 'train_stats')
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)
    cached_path = os.path.join(fold_path, 'cached')
    if os.path.exists(cached_path):
        encounter_dict = {}
        print('Processing encodedVisit.csv')
        encounter_dict = process_hospital_detail(hospital_detail_file, encounter_dict)
        encounter_dict = processs_demographics(hospital_detail_file, encounter_dict)
        print('Processing eventStatistics.csv')
        encounter_dict = processs_vital_signs(vital_signs_file, encounter_dict)
        print('Processing HospDiagnosis.csv')
        encounter_dict = process_HospDiagnosis(HospDiagnosis_file, icd10toicd9dx_file, multi_dx_file, encounter_dict)
        print('Processing HospProcedures.csv')
        encounter_dict = process_HospProcedures(HospProcedures_file, icd10toicd9proc_file, multi_proc_file, encounter_dict)
        key_list, enc_features_list, key_list_live, key_list_death = get_encounter_features(encounter_dict, skip_duplicate=False, min_num_codes=1, max_num_codes=30)

        print("len(key_list_death): ", len(key_list_death))
        print("len(key_list_live): ", len(key_list_live))

        key_live_train, key_live_valid, key_live_test = select_train_valid_test(key_list_live, random_seed=fold)
        key_death_train, key_death_valid, key_death_test = select_train_valid_test(key_list_death, random_seed=fold)
        key_expired_train = key_live_train + key_death_train

        count_conditional_prob_dp_orderset_bylabel(enc_features_list, stats_path, set(key_expired_train), label="expired")

        train_live_enc_features = add_sparse_prior_guide_dp_orderset_bylabel(enc_features_list, stats_path, set(key_live_train), max_num_codes=30, label="expired")
        train_death_enc_features = add_sparse_prior_guide_dp_orderset_bylabel(enc_features_list, stats_path, set(key_death_train), max_num_codes=30, label="expired")
        valid_live_enc_features = add_sparse_prior_guide_dp_orderset_bylabel(enc_features_list, stats_path, set(key_live_valid), max_num_codes=30, label="expired")
        valid_death_enc_features = add_sparse_prior_guide_dp_orderset_bylabel(enc_features_list, stats_path, set(key_death_valid), max_num_codes=30, label="expired")
        test_live_enc_features = add_sparse_prior_guide_dp_orderset_bylabel(enc_features_list, stats_path, set(key_live_test), max_num_codes=30, label="expired")
        test_death_enc_features = add_sparse_prior_guide_dp_orderset_bylabel(enc_features_list, stats_path, set(key_death_test), max_num_codes=30, label="expired")

        train_expired_enc_features = train_live_enc_features + train_death_enc_features
        validation_expired_enc_features = valid_live_enc_features + valid_death_enc_features
        test_expired_enc_features = test_live_enc_features + test_death_enc_features

        train_expired_dataset = convert_features_to_tensors(train_expired_enc_features)
        validation_expired_dataset = convert_features_to_tensors(validation_expired_enc_features)
        test_expired_dataset = convert_features_to_tensors(test_expired_enc_features)

        torch.save(train_expired_dataset, os.path.join(cached_path, 'train_dataset_custom_expired.pt'))
        torch.save(validation_expired_dataset, os.path.join(cached_path, 'valid_dataset_custom_expired.pt'))
        torch.save(test_expired_dataset, os.path.join(cached_path, 'test_dataset_custom_expired.pt'))

        train_expired_prior_guide = get_prior_guide(train_expired_enc_features)
        validation_expired_prior_guide = get_prior_guide(validation_expired_enc_features)
        test_expired_prior_guide = get_prior_guide(test_expired_enc_features)

        torch.save(train_expired_prior_guide, os.path.join(cached_path, 'train_priors_custom_expired.pt'))
        torch.save(validation_expired_prior_guide, os.path.join(cached_path, 'valid_priors_custom_expired.pt'))
        torch.save(test_expired_prior_guide, os.path.join(cached_path, 'test_priors_custom_expired.pt'))

        print("custom label=expired dataset done")

        if flag == "expired":
            train_dataset = train_expired_dataset
            validation_dataset = validation_expired_dataset
            test_dataset = test_expired_dataset

            train_prior_guide = train_expired_prior_guide
            validation_prior_guide = validation_expired_prior_guide
            test_prior_guide = test_expired_prior_guide

    return (
    [train_dataset, validation_dataset, test_dataset], [train_prior_guide, validation_prior_guide, test_prior_guide])


if __name__ == "__main__":
    datasets, prior_guides = get_mimiciv_datasets("data/mimic4-3.0", 50)
