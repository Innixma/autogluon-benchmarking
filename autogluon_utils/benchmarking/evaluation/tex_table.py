import pandas as pd
import numpy as np

# Example usage:  x = tex_table(df, textable_file="testlatextable.txt", bold='min')

def tex_table(df, textable_file, bold = None, nan_char = " x ", max_digits = 4):
    """ This function is only intended for fully numerical tables (dataset x frameworks comparison).
        Datasets should be row indices of df rather than a column.
    Args:
        df = DataFrame
        textable_file = path to output file
        bold = 'min' or = 'max' (if df only contains numbers), or = None for no bolding.
        nan_char replaces NaN in LaTex table
        max_digits = Maximum number of digits to show in each cell. 
    """
    if bold is not None:
        if bold == 'min':
            best_row_vals = df.min(axis=1)
            # best_cols = df.idxmin(axis=1)
        elif bold == 'max':
            best_row_vals = df.max(axis=1)
            # best_cols = df.idxmax(axis=0)
        else:
            raise ValueError("unknown bold option")
        best_cols = []
        for i in df.index:
            row_best_cols = list(df.columns[np.abs(df.loc[i] - best_row_vals[i]) < 1e-5])
            best_cols.append(row_best_cols)
            if len(row_best_cols) <= 0:
                raise ValueError("no row value matches best row value")
        
    # SHIFT_FACTOR = 100
    # df = df * SHIFT_FACTOR
    # df = df.round(num_decimals)
    # df = df / SHIFT_FACTOR
    max_int = int(df.max(numeric_only=True).max())
    max_digits = max(max_digits, len(str(max_int))) # make sure we don't truncate values before decimal
    df = df.astype('str')
    df = df.replace("nan", nan_char)
    df = df.applymap(lambda x: x[:max_digits])
    
    print(df.columns)
    if bold is not None:
        ind = 0
        for i in df.index: # bold best value:
            if len(best_cols[ind]) > 0:
                for col_name in best_cols[ind]:
                    df.at[i,col_name] = "\\textbf{" + df.at[i,col_name]  + "}"
            ind += 1
    
    df.reset_index(inplace=True) # set dataset indices as first column
    df.rename(columns={'dataset':'Dataset'}, inplace=True)
    cols = list(df.columns)
    df.columns = ['\\textbf{'+col+'}' for col in cols]
    textab = df.to_latex(escape=True, index=False, column_format = 'l'+'c'*(len(df.columns)-1))
    textab = textab.replace("\\textbackslash textbf", "\\textbf")
    textab = textab.replace("\\{", "{")
    textab = textab.replace("\\}", "}")
    
    with open(textable_file,'w') as tf:
        tf.write(textab)
        print("saved tex table to: %s" % textable_file)
    return textab

