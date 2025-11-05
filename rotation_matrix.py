# Rotation matrix builder (retry)
# This version is robust and avoids importing optional helpers that might not exist.
# It will produce the same CSV outputs in /mnt/data and print concise summaries.
import pandas as pd
from pathlib import Path

# --- Embedded data ---
rows = [
    ("Mango", "Fruits", 0.4468, 0.2458, 0.0757, 0.0044, 0.0261),
    ("Papaya", "Fruits", 0.4754, 0.2748, 0.1778, 0.0172, 0.0398),
    ("Orange", "Fruits", 0.4707, 0.2537, 0.1280, 0.0083, 0.0254),
    ("Banana", "Fruits", 0.4195, 0.1963, 0.1140, 0.0088, 0.0801),
    ("Guava", "Fruits", 0.5156, 0.3102, 0.1791, 0.0120, 0.0570),
    ("Watermelon", "Fruits", 0.3111, 0.0706, 0.0833, 0.0089, 0.0152),
    ("Apple", "Fruits", 0.3710, 0.0884, 0.0776, 0.0045, 0.0133),
    ("Avocado", "Fruits", 0.4674, 0.2455, 0.0511, 0.0026, 0.0357),
    ("Durian", "Fruits", 0.4516, 0.2483, 0.0275, 0.0016, 0.0203),
    ("Corn", "Starchy staples", 0.3908, 0.1535, 0.1214, 0.0113, 0.4179),
    ("Potato", "Starchy staples", 0.4782, 0.3053, 0.1246, 0.0113, 0.0934),
    ("Tofu", "Pulses, nuts, and seeds", 0.5211, 0.3471, 0.1052, 0.0188, 0.1026),
    ("Tempeh", "Pulses, nuts, and seeds", 0.5391, 0.3946, 0.1115, 0.0201, 0.2248),
    ("Peanuts", "Pulses, nuts, and seeds", 0.4650, 0.4268, 0.0546, 0.0031, 0.2678),
    ("Chickpeas", "Pulses, nuts, and seeds", 0.5153, 0.3286, 0.1404, 0.0125, 0.3980),
    ("Pumpkin", "Vegetables", 0.5889, 0.4766, 0.0579, 0.0030, 0.0338),
    ("Spinach", "Vegetables", 0.9032, 0.9346, 0.0859, 0.0043, 0.0362),
    ("Tomatoes", "Vegetables", 0.5816, 0.4394, 0.1039, 0.0061, 0.0387),
    ("Long bean", "Vegetables", 0.5616, 0.4127, 0.0821, 0.0047, 0.3634),
    ("Cabbage", "Vegetables", 0.6376, 0.5007, 0.0791, 0.0043, 0.0341),
    ("Eggplant", "Vegetables", 0.3967, 0.1731, 0.0597, 0.0035, 0.0217),
    ("Cucumber", "Vegetables", 0.4306, 0.2272, 0.1058, 0.0084, 0.0188),
    ("Egg", "Animal-source foods", 0.5837, 0.4851, 0.0343, 0.0017, 0.0217),
    ("Beef", "Animal-source foods", 0.5968, 0.5424, 0.0038, 0.4468, 0.0241),
    ("Lamb", "Animal-source foods", 0.5941, 0.5332, 0.0088, 0.0005, 0.0242),
    ("Pork", "Animal-source foods", 0.5840, 0.5233, 0.0165, 0.0008, 0.3743),
    ("Chicken", "Animal-source foods", 0.5533, 0.4336, 0.0249, 0.0013, 0.0572),
]
cols = ["Food_Name", "Food_Group", "Nut_Val", "Nut_Den", "Sust", "Env_Imp", "Afford"]
df = pd.DataFrame(rows, columns=cols)

# Compute group mean environmental impact
group_env = df.groupby("Food_Group", sort=True)["Env_Imp"].mean().reset_index().rename(columns={"Env_Imp":"Env_Mean"})

# Default priors (editable)
groups = list(group_env["Food_Group"])

s_g_default = {
    "Pulses, nuts, and seeds": 1.0,
    "Vegetables": 0.6,
    "Fruits": 0.2,
    "Starchy staples": 0.0,
    "Animal-source foods": -0.5,
}
u_h_default = {
    "Vegetables": 1.0,
    "Starchy staples": 0.8,
    "Fruits": 0.5,
    "Pulses, nuts, and seeds": 0.2,
    "Animal-source foods": 0.0,
}

# Ensure defaults present
for g in groups:
    s_g_default.setdefault(g, 0.0)
    u_h_default.setdefault(g, 0.0)

base = 0.4
lam = 3.0

# Compute G matrix
env_dict = dict(zip(group_env["Food_Group"], group_env["Env_Mean"]))
s_g = s_g_default.copy()
u_h = u_h_default.copy()

G = pd.DataFrame(index=groups, columns=groups, dtype=float)
for g in groups:
    for h in groups:
        env_g = env_dict[g]
        G.loc[g, h] = base * s_g[g] * (1.0 - lam * env_g) * u_h[h]

# Expand to crop x crop R matrix
crop_list = df["Food_Name"].tolist()
crop_group = dict(zip(df["Food_Name"], df["Food_Group"]))
R = pd.DataFrame(index=crop_list, columns=crop_list, dtype=float)
for c in crop_list:
    for c2 in crop_list:
        R.loc[c, c2] = G.loc[crop_group[c], crop_group[c2]]

# Save outputs to current directory
outdir = Path("./rotation_data")
outdir.mkdir(parents=True, exist_ok=True)
group_env.to_csv(outdir / "group_env_means.csv", index=False)
G.to_csv(outdir / "rotation_group_matrix.csv")
R.to_csv(outdir / "rotation_crop_matrix.csv")

# Print concise summaries and file paths
print("Group Env Means:")
print(group_env.to_string(index=False))
print("\nUsed s_g (soil/legacy):", s_g)
print("Used u_h (beneficiary):", u_h)
print(f"\nParameters: base={base}, lambda={lam}\n")
print("Rotation G (group->group) matrix (rows=prev group, cols=next group):")
print(G.to_string(float_format=lambda x: f"{x:0.4f}"))
print("\nSaved CSV files to ./rotation_data/:")
print(" - rotation_data/group_env_means.csv")
print(" - rotation_data/rotation_group_matrix.csv")
print(" - rotation_data/rotation_crop_matrix.csv")

# Also show top-left corner of R for quick verification
print("\nR matrix sample (first 8 crops rows x first 8 crops cols):")
print(R.iloc[:8, :8].to_string(float_format=lambda x: f"{x:0.4f}"))
