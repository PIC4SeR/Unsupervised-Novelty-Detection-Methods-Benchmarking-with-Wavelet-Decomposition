
def print_result_table(table: dict, preamble: str, out: str):
    with open(out, 'w') as f:
        # copy preambolo from the original table
        with open(preamble, 'r') as f_preamble:
            for line in f_preamble:
                f.write(line)
            f.write('\n')
        # write the table
        for model in table.keys():
            string = '\multirow{4}{*}{' + str(model) + '} &  None & ' + str(table[model]['original']['n_red_features']) + ' & ' + str(table[model]['original']['Feat_red_factor']) + ' & ' + str(table[model]['original']['Variance']) + ' & ' + str(table[model]['original']['Mean_1']) + ' & ' + str(table[model]['original']['Mean_2']) + ' & ' + str(table[model]['original']['Reactivity']) + ' &  ' + str(table[model]['original']['Inference_time']) + r' \\'
            f.write(string + '\n')
            for reduction in table[model].keys():
                if reduction != 'original':
                    #f.write('\cline{2-8}\n')
                    string = ' & ' + str(reduction).replace('_augmented','A').replace('_reduced','R') + ' & ' + str(table[model][reduction]['n_red_features']) + ' & ' + str(table[model][reduction]['Feat_red_factor']) + ' & ' + str(table[model][reduction]['Variance']) + ' & ' + str(table[model][reduction]['Mean_1']) + ' & ' + str(table[model][reduction]['Mean_2']) + ' & ' + str(table[model][reduction]['Reactivity']) + ' &  ' + str(table[model][reduction]['Inference_time']) + r' \\'
                    f.write(string + '\n')
            if model != list(table.keys())[-1]:
                f.write('\hline\n')
            else:
                f.write('\\bottomrule\n')
        f.write('\end{tabular}\n')
        f.write('\end{table*}\n')
        f.write('\endgroup\n')