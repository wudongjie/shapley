import shapley
import pandas as pd
import argparse
import models
import time
import gc
import os
import csv

def shapley_value_for_all(model, verbose, step):
    shapley_value = list()
    for i in range(model.num_factors + 1):
        start_time = time.time()
        shap10000 = shapley.Shapley(i, model=model, verbose=verbose,
                                    method="sampling", steps=step)
        value = shap10000.shapley_value
        if verbose:
            print(" --- %s seconds ---" % (time.time() - start_time))
        shapley_value.append(value)
        gc.collect()
    return shapley_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="verbose computation",
                        action="store_true")
    parser.add_argument("-i", "--input", help="add the input csv file")
    parser.add_argument("-o", "--output", help="the csv file you want to output (default result.csv)", default="result.csv")
    parser.add_argument("-m", "--model",
                        help="specify the model, currently support 'linear' or 'log-linear', default: linear",
                        default="linear")
    parser.add_argument("-s", "--step", help="steps for computing each Shapley value (default 10,000)",
                        default=10000, type=int)
    args = parser.parse_args()
    assert args.input, "You didn't provide the input file!"
    assert os.path.exists(args.input) == 1, "The input file does not exist!"
    assert args.model == 'linear' or args.model == 'log-linear', "The model must be either linear or log-linear"
    assert type(args.step) == int, "Steps must be integer!"
    df = pd.read_csv(args.input, engine='c')
    df = df.dropna()
    print("Dataset Loaded!")
    model_selected = models.Models(df, model=args.model)
    coefs = model_selected.coefs
    varnames = model_selected.varnames
    shap = shapley_value_for_all(model_selected, args.verbose, args.step)
    with open(args.output, 'w') as myfile:
        wr = csv. writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(['Variable Name'] + varnames[1:] + ['Residuals'])
        wr.writerow(['Coefficients'] + list(coefs))
        wr.writerow(['Shapley Value'] + shap)

if __name__ == "__main__":
    main()
