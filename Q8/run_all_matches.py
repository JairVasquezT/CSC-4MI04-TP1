import subprocess
import csv
import os

csv_file = "resultats_matching_parametres.csv"
results_for_csv = []

# Formato ORB: (det, est, ratio, nfeatures, nlevels, fast_threshold)
# Formato KAZE: (det, est, ratio, threshold, octaves, upright_int) 
# (upright_int: 0 para False, 1 para True)
configs = [
    # --- ORB ---
    ("orb", "ratiotest", 0.7, 500, 8, 20),
    ("orb", "ratiotest", 0.85, 1200, 12, 20),
    ("orb", "crosscheck", 0.0, 300, 15, 40),
    ("orb", "flann", 0.7, 500, 8, 20),
    
    # --- KAZE ---
    ("kaze", "ratiotest", 0.7, 0.001, 4, 0),
    ("kaze", "ratiotest", 0.5, 0.001, 4, 0),
    ("kaze", "ratiotest", 0.6, 0.004, 4, 0),
    ("kaze", "crosscheck", 0.0, 0.005, 4, 1), 
    ("kaze", "flann", 0.7, 0.001, 4, 0)
]

print(f"{'Detector':<10} | {'Stratégie':<12} | {'Matches':<8} | {'Temps'}")
print("-" * 50)

for det, est, rat, p1, p2, p3 in configs:
    if det == "orb":
        # p1=nf, p2=nl, p3=ft
        cmd = ["python3", "Features_Match.py", det, est, "--no-show", 
               "--ratio", str(rat), "--nfeatures", str(p1), "--nlevels", str(p2), 
               "--scale", "1.2", "--fast-threshold", str(p3)]
        params = f"nf={p1},nl={p2},ft={p3}"
    else:
        # p1=th, p2=oct, p3=upright
        cmd = ["python3", "Features_Match.py", det, est, "--no-show", 
               "--ratio", str(rat), "--kaze-threshold", str(p1), 
               "--kaze-octaves", str(p2), "--kaze-layers", "4"]
        if p3 == 1: cmd.append("--upright") 
        params = f"th={p1},o={p2},u={p3}"
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in res.stdout.split('\n'):
            if "RESULT_DATA|" in line:
                parts = line.split('|')
                d, s, m, t = parts[1], parts[2], parts[3], parts[4]
                print(f"{d.upper():<10} | {s:<12} | {m:<8} | {t}s")
                results_for_csv.append([d, s, rat, params, m, t])
                break
    except Exception as e:
        print(f"{det.upper():<10} | {est:<12} | ERROR")

# --- GUARDAR EN CSV ---
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Detecteur', 'Strategie', 'Ratio', 'Parametres', 'Nb_Matches', 'Temps'])
    writer.writerows(results_for_csv)