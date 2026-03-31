import os
import shutil
import csv
from src.utils.config import load_config

# ==== PARAMETRY ====
cfg = load_config("configs/config_list.yaml")
path = os.path.join("results", cfg["experiment"]["name"])
csv_path = os.path.join(path, "prediction/metrics_sorted.csv")
image_column = "filename"  # název sloupce v CSV
x = 10  # kolik řádků vzít

train_ratio = 0.8  # poměr train/val

# výstupní složky
best_base = os.path.join(path, "prediction_selection/best", cfg["predict"]["prefix"])
worst_base = os.path.join(path, "prediction_selection/worst", cfg["predict"]["prefix"])

# ==== NAČTENÍ CSV ====
rows = []

with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# první a poslední x
best_rows = rows[:x]
worst_rows = rows[-x:]

# ==== FUNKCE PRO KOPÍROVÁNÍ ====
def copy_files(rows, base_dir):
    os.makedirs(base_dir, exist_ok=True)

    for row in rows:
        filename = row[image_column]

        src = os.path.join(path, "prediction", filename)

        if not os.path.exists(src):
            print(f"Soubor neexistuje: {src}")
            continue

        filename = os.path.basename(src)
        dst = os.path.join(base_dir, filename)

        # copy originálu
        shutil.copy2(src, dst)

        # g_ varianta
        g_filename = "g_" + filename
        g_src = os.path.join(path, "prediction", g_filename)
        g_dst = os.path.join(base_dir, g_filename)

        if os.path.exists(g_src):
            shutil.copy2(g_src, g_dst)
        else:
            print(f"g_ soubor neexistuje: {g_src}")

# ==== SPUŠTĚNÍ ====
copy_files(best_rows, best_base)
copy_files(worst_rows, worst_base)

print("Hotovo.")