import pickle
import collections
import argparse
import os
import torch


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    vocab['PAD'] = len(vocab)
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = len(vocab)
    return vocab


def build_trees_diag(vocab_file, graph_file, out_file):
    """
       Get the diagnosis ontology structure and transform the input data
       :param graph_file: The path to 'ccs_multi_dx_tool_2015.csv'
       :param vocab_file: The path to diagnosis code mapping file 'diag_vocab.txt' extracted by 'process_mimic.py'
       :param out_file: The output path
       :return:  new_types：diagnosis code mapping file add ancestors
       """
    print('Read Saved data dictionary')
    types = load_vocab(vocab_file)

    start_set = set(types.keys())
    hit_list = []

    cat1count = 0
    cat2count = 0
    cat3count = 0
    cat4count = 0

    graph = open(graph_file, 'r')
    _ = graph.readline()

    # add ancestors to dictionary
    for line in graph:
        tokens = line.strip().split(',')
        icd9 = tokens[0][1:-1].strip()
        cat1 = tokens[1][1:-1].strip()
        desc1 = 'A_L1_' + cat1
        cat2 = tokens[3][1:-1].strip()
        desc2 = 'A_L2_' + cat2
        cat3 = tokens[5][1:-1].strip()
        desc3 = 'A_L3_' + cat3
        cat4 = tokens[7][1:-1].strip()
        desc4 = 'A_L4_' + cat4

        icd9 = 'D_' + icd9

        if icd9 not in types:
            continue
        else:
            hit_list.append(icd9)

        if desc1 not in types:
            cat1count += 1
            types[desc1] = len(types)

        if len(cat2) > 0:
            if desc2 not in types:
                cat2count += 1
                types[desc2] = len(types)
        if len(cat3) > 0:
            if desc3 not in types:
                cat3count += 1
                types[desc3] = len(types)
        if len(cat4) > 0:
            if desc4 not in types:
                cat4count += 1
                types[desc4] = len(types)
    graph.close()

    # add root_code
    types['A_ROOT'] = len(types)
    root_code = types['A_ROOT']

    miss_set = start_set - set(hit_list)
    print('missing code: {}'.format(len(miss_set)))
    print('cat1count:', cat1count)
    print('cat2count:', cat2count)
    print('cat3count:', cat3count)
    print('cat4count:', cat4count)

    five_map = {}
    four_map = {}
    three_map = {}
    two_map = {}
    one_map = dict([(types[icd], [types[icd], root_code]) for icd in miss_set])

    graph = open(graph_file, 'r')
    graph.readline()

    for line in graph:
        tokens = line.strip().split(',')
        icd9 = tokens[0][1:-1].strip()
        cat1 = tokens[1][1:-1].strip()
        desc1 = 'A_L1_' + cat1
        cat2 = tokens[3][1:-1].strip()
        desc2 = 'A_L2_' + cat2
        cat3 = tokens[5][1:-1].strip()
        desc3 = 'A_L3_' + cat3
        cat4 = tokens[7][1:-1].strip()
        desc4 = 'A_L4_' + cat4

        icd9 = 'D_' + icd9

        if icd9 not in types:
            continue

        icd_code = types[icd9]

        if len(cat4) > 0:
            code4 = types[desc4]
            code3 = types[desc3]
            code2 = types[desc2]
            code1 = types[desc1]
            five_map[icd_code] = [icd_code, root_code, code1, code2, code3, code4]
        elif len(cat3) > 0:
            code3 = types[desc3]
            code2 = types[desc2]
            code1 = types[desc1]
            four_map[icd_code] = [icd_code, root_code, code1, code2, code3]

        elif len(cat2) > 0:
            code2 = types[desc2]
            code1 = types[desc1]
            three_map[icd_code] = [icd_code, root_code, code1, code2]

        else:
            code1 = types[desc1]
            two_map[icd_code] = [icd_code, root_code, code1]

    # Now we re-map the integers to all medical leaf codes.
    new_five_map = collections.OrderedDict()
    new_four_map = collections.OrderedDict()
    new_three_map = collections.OrderedDict()
    new_two_map = collections.OrderedDict()
    new_one_map = collections.OrderedDict()
    new_types = collections.OrderedDict()
    types_reverse = dict([(v, k) for k, v in types.items()])

    srcs = []
    dsts = []
    vals = []

    code_count = 0
    for icdCode, ancestors in five_map.items():
        new_types[types_reverse[icdCode]] = code_count
        new_five_map[code_count] = [code_count] + ancestors[1:]
        code_count += 1
        lst = [code_count] + ancestors[1:]
        for i in range(len(lst) - 1):
            src = lst[i] - 1
            dst = lst[i + 1] - 1
            srcs.append(src)
            dsts.append(dst)
            vals.append(1.0)
            srcs.append(dst)
            dsts.append(src)
            vals.append(1.0)

    for icdCode, ancestors in four_map.items():
        new_types[types_reverse[icdCode]] = code_count
        new_four_map[code_count] = [code_count] + ancestors[1:]
        code_count += 1
        lst = [code_count] + ancestors[1:]
        for i in range(len(lst) - 1):
            src = lst[i] - 1
            dst = lst[i + 1] - 1
            srcs.append(src)
            dsts.append(dst)
            vals.append(1.0)
            srcs.append(dst)
            dsts.append(src)
            vals.append(1.0)

    for icdCode, ancestors in three_map.items():
        new_types[types_reverse[icdCode]] = code_count
        new_three_map[code_count] = [code_count] + ancestors[1:]
        code_count += 1
        lst = [code_count] + ancestors[1:]
        for i in range(len(lst) - 1):
            src = lst[i] - 1
            dst = lst[i + 1] - 1
            srcs.append(src)
            dsts.append(dst)
            vals.append(1.0)
            srcs.append(dst)
            dsts.append(src)
            vals.append(1.0)

    for icdCode, ancestors in two_map.items():
        new_types[types_reverse[icdCode]] = code_count
        new_two_map[code_count] = [code_count] + ancestors[1:]
        code_count += 1
        lst = [code_count] + ancestors[1:]
        for i in range(len(lst) - 1):
            src = lst[i] - 1
            dst = lst[i + 1] - 1
            srcs.append(src)
            dsts.append(dst)
            vals.append(1.0)
            srcs.append(dst)
            dsts.append(src)
            vals.append(1.0)

    for icdCode, ancestors in one_map.items():
        new_types[types_reverse[icdCode]] = code_count
        new_one_map[code_count] = [code_count] + ancestors[1:]
        code_count += 1
        lst = [code_count] + ancestors[1:]
        for i in range(len(lst) - 1):
            src = lst[i] - 1
            dst = lst[i + 1] - 1
            srcs.append(src)
            dsts.append(dst)
            vals.append(1.0)
            srcs.append(dst)
            dsts.append(src)
            vals.append(1.0)

    pickle.dump(new_types, open(os.path.join(out_file, 'tree.diag.inputs.dict'), 'wb'), -1)
    pickle.dump(new_five_map, open(os.path.join(out_file, 'tree.diag.level5.pk'), 'wb'), -1)
    pickle.dump(new_four_map, open(os.path.join(out_file, 'tree.diag.level4.pk'), 'wb'), -1)
    pickle.dump(new_three_map, open(os.path.join(out_file, 'tree.diag.level3.pk'), 'wb'), -1)
    pickle.dump(new_two_map, open(os.path.join(out_file, 'tree.diag.level2.pk'), 'wb'), -1)
    pickle.dump(new_one_map, open(os.path.join(out_file, 'tree.diag.level1.pk'), 'wb'), -1)
    print('len(new_types):', len(new_types))
    print('len(types_reverse):', len(types_reverse))
    print('len(new_five_map):', len(new_five_map))
    print('len(new_four_map):', len(new_four_map))
    print('len(new_three_map):', len(new_three_map))
    print('len(new_two_map):', len(new_two_map))
    print('len(new_one_map):', len(new_one_map))  # print(new_four_map)

    return new_types


def build_trees_proc(vocab_file, graph_file, out_file):
    """
       Get the procedure ontology structure and transform the input data
       :param graph_file: The path to 'ccs_multi_pr_tool_2015.csv'
       :param vocab_file: The path to procedure code mapping file 'proc_vocab.txt' extracted by 'process_mimic.py'
       :param out_file: The output path
       :return:  new_types：procedure code mapping file add ancestors
       """
    print('Read Saved data dictionary')
    types = load_vocab(vocab_file)

    start_set = set(types.keys())
    hit_list = []

    cat1count = 0
    cat2count = 0
    cat3count = 0

    graph = open(graph_file, 'r')
    _ = graph.readline()

    # add ancestors to dictionary
    for line in graph:
        tokens = line.strip().split(',')
        icd9 = tokens[0][1:-1].strip()
        cat1 = tokens[1][1:-1].strip()
        desc1 = 'B_L1_' + cat1
        cat2 = tokens[3][1:-1].strip()
        desc2 = 'B_L2_' + cat2
        cat3 = tokens[5][1:-1].strip()
        desc3 = 'B_L3_' + cat3

        icd9 = 'P_' + icd9

        if icd9 not in types:
            continue
        else:
            hit_list.append(icd9)

        if desc1 not in types:
            cat1count += 1
            types[desc1] = len(types)

        if len(cat2) > 0:
            if desc2 not in types:
                cat2count += 1
                types[desc2] = len(types)
        if len(cat3) > 0:
            if desc3 not in types:
                cat3count += 1
                types[desc3] = len(types)

    graph.close()

    # add root_code
    types['A_ROOT'] = len(types)
    root_code = types['A_ROOT']
    miss_set = start_set - set(hit_list)
    print('missing code: {}'.format(len(miss_set)))
    print('cat1count:', cat1count)
    print('cat2count:', cat2count)
    print('cat3count:', cat3count)

    four_map = {}
    three_map = {}
    two_map = {}
    one_map = dict([(types[icd], [types[icd], root_code]) for icd in miss_set])

    graph = open(graph_file, 'r')
    graph.readline()

    for line in graph:
        tokens = line.strip().split(',')
        icd9 = tokens[0][1:-1].strip()
        cat1 = tokens[1][1:-1].strip()
        desc1 = 'B_L1_' + cat1
        cat2 = tokens[3][1:-1].strip()
        desc2 = 'B_L2_' + cat2
        cat3 = tokens[5][1:-1].strip()
        desc3 = 'B_L3_' + cat3

        icd9 = 'P_' + icd9

        if icd9 not in types:
            continue

        icd_code = types[icd9]

        if len(cat3) > 0:
            code3 = types[desc3]
            code2 = types[desc2]
            code1 = types[desc1]
            four_map[icd_code] = [icd_code, root_code, code1, code2, code3]

        elif len(cat2) > 0:
            code2 = types[desc2]
            code1 = types[desc1]
            three_map[icd_code] = [icd_code, root_code, code1, code2]

        else:
            code1 = types[desc1]
            two_map[icd_code] = [icd_code, root_code, code1]

    # Now we re-map the integers to all medical leaf codes.
    new_four_map = collections.OrderedDict()
    new_three_map = collections.OrderedDict()
    new_two_map = collections.OrderedDict()
    new_one_map = collections.OrderedDict()
    new_types = collections.OrderedDict()
    types_reverse = dict([(v, k) for k, v in types.items()])

    srcs = []
    dsts = []
    vals = []

    code_count = 0
    for icdCode, ancestors in four_map.items():
        new_types[types_reverse[icdCode]] = code_count
        new_four_map[code_count] = [code_count] + ancestors[1:]
        code_count += 1
        lst = [code_count] + ancestors[1:]
        for i in range(len(lst) - 1):
            src = lst[i] - 1
            dst = lst[i + 1] - 1
            srcs.append(src)
            dsts.append(dst)
            vals.append(1.0)
            srcs.append(dst)
            dsts.append(src)
            vals.append(1.0)

    for icdCode, ancestors in three_map.items():
        new_types[types_reverse[icdCode]] = code_count
        new_three_map[code_count] = [code_count] + ancestors[1:]
        code_count += 1
        lst = [code_count] + ancestors[1:]
        for i in range(len(lst) - 1):
            src = lst[i] - 1
            dst = lst[i + 1] - 1
            srcs.append(src)
            dsts.append(dst)
            vals.append(1.0)
            srcs.append(dst)
            dsts.append(src)
            vals.append(1.0)

    for icdCode, ancestors in two_map.items():
        new_types[types_reverse[icdCode]] = code_count
        new_two_map[code_count] = [code_count] + ancestors[1:]
        code_count += 1
        lst = [code_count] + ancestors[1:]
        for i in range(len(lst) - 1):
            src = lst[i] - 1
            dst = lst[i + 1] - 1
            srcs.append(src)
            dsts.append(dst)
            vals.append(1.0)
            srcs.append(dst)
            dsts.append(src)
            vals.append(1.0)

    for icdCode, ancestors in one_map.items():
        new_types[types_reverse[icdCode]] = code_count
        new_one_map[code_count] = [code_count] + ancestors[1:]
        code_count += 1
        lst = [code_count] + ancestors[1:]
        for i in range(len(lst) - 1):
            src = lst[i] - 1
            dst = lst[i + 1] - 1
            srcs.append(src)
            dsts.append(dst)
            vals.append(1.0)
            srcs.append(dst)
            dsts.append(src)
            vals.append(1.0)


    pickle.dump(new_types, open(os.path.join(out_file, 'tree.proc.inputs.dict'), 'wb'), -1)
    pickle.dump(new_four_map, open(os.path.join(out_file, 'tree.proc.level4.pk'), 'wb'), -1)
    pickle.dump(new_three_map, open(os.path.join(out_file, 'tree.proc.level3.pk'), 'wb'), -1)
    pickle.dump(new_two_map, open(os.path.join(out_file, 'tree.proc.level2.pk'), 'wb'), -1)
    pickle.dump(new_one_map, open(os.path.join(out_file, 'tree.proc.level1.pk'), 'wb'), -1)
    print('len(new_types):', len(new_types))
    print('len(types_reverse):', len(types_reverse))
    print('len(new_four_map):', len(new_four_map))
    print('len(new_three_map):', len(new_three_map))
    print('len(new_two_map):', len(new_two_map))
    print('len(new_one_map):', len(new_one_map))  # print(new_four_map)

    return new_types


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output", type=str, required=False, default='processed_data/mimic4/')
    parser.add_argument("--diag_vocab", type=str, required=False, default='processed_data/mimic4/diag_vocab.txt')
    parser.add_argument("--diag_ccs_multi_file", type=str, default='ccs/ccs_multi_dx_tool_2015.csv')
    parser.add_argument("--proc_vocab", type=str, required=False, default='processed_data/mimic4/proc_vocab.txt')
    parser.add_argument("--proc_ccs_multi_file", type=str, required=False, default='ccs/ccs_multi_pr_tool_2015.csv')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    build_trees_diag(args.diag_vocab, args.diag_ccs_multi_file, args.output)
    build_trees_proc(args.proc_vocab, args.proc_ccs_multi_file, args.output)
