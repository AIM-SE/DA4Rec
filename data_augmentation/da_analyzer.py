import os
import argparse
import pickle
import itertools
import xlwt
from tabulate import tabulate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_paths", type=str, default=None)
    parser.add_argument("--online_aug", action="store_true", default=False)

    args = parser.parse_args()

    result_path_list = os.listdir(args.result_paths)

    wb = xlwt.Workbook()
    for result_path in result_path_list:
        current_root = os.path.join(args.result_paths, result_path)

        if os.path.isfile(current_root):
            continue

        result_list = os.listdir(current_root)

        metrics = ["hit", "mrr", "ndcg"]
        topk = ["@1", "@3", "@5", "@10", "@20", "@50"]
        heads = []

        for metric, k in itertools.product(metrics, topk):
            heads.append(metric + k)

        heads.append("number of instances")
        heads.append("seed")
        heads.append("train_epoch_cost")
        heads.append("valid_epoch_cost")

        if args.online_aug:
            heads += ["insert", "replace", "mask", "delete", "subset-split", "skip"]

        table = [heads]

        for result in result_list:
            with open(os.path.join(current_root, result), "rb") as f:
                performance = pickle.load(f)
                line = [result]
                for metric in heads:
                    if performance[metric] is not None:
                        line.append(performance[metric])
                table.append(line)

        table[0].insert(0, result_path)
        print(tabulate(table, headers="firstrow"))
        print("\n")

        # export to xls
        sheet = wb.add_sheet(result_path[7:] if len(result_path)>7 else result_path)
        x_dims = len(table)
        y_dims = len(table[0])
        for y in range(y_dims):
            for x in range(x_dims):
                sheet.write(x, y, table[x][y])
    wb.save("export.xls")
