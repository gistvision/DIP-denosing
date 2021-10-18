from invoke import task
import pandas as pd

@task
def showtable(c, csv_dir, prefix = ""):
    import os
    csv_list = os.listdir(csv_dir)
    csv_list = sorted(csv_list)   

    print_result = lambda i, tmp: print(i[:-4], "\t\t\t", ("optimal stopping : %.2f,\t" + "%.2f/%.2f \t| ZCSC : %.2f, \t %.2f/%.2f | STE %.2f/%.2f") %
                                        (tmp["max_step"], tmp["max_psnr"], tmp["max_lpips"] * 10,
                                         tmp["final_ep"], tmp["final_psnr"],  tmp["final_lpips"] * 10, tmp["final_psnr_avg"], tmp["final_lpips_avg"] * 10,))

    for i in csv_list:
        tmp = pd.read_csv(os.path.join(csv_dir, i))
        tmp_mean = tmp.mean()
        print_result(i, tmp_mean)
