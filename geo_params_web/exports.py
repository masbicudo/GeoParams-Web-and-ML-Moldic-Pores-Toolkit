from libs import data

# ---------------------------------------------------------------------
# Load raw interaction data (may include personal fields)
# ---------------------------------------------------------------------
df = data.load_data_from_files(debug=False)

# ---------------------------------------------------------------------
# Remove canceled interactions
# ---------------------------------------------------------------------
df = df[~df["canceled"]]

# ---------------------------------------------------------------------
# Remove developer test entries (based on name, before dropping it)
# ---------------------------------------------------------------------
if "name" in df.columns:
    df = df[~df["name"].str.contains("Test", case=False, na=False)]

# ---------------------------------------------------------------------
# Basic sanity bounds for click coordinates
# ---------------------------------------------------------------------
df = df[
    (0 <= df["clicked_x"]) & (df["clicked_x"] < 256)
    & (0 <= df["clicked_y"]) & (df["clicked_y"] < 256)
]

# ---------------------------------------------------------------------
# Sort for reproducibility (not required, but nice)
# ---------------------------------------------------------------------
df = df.sort_values(by=["experience"])

# ---------------------------------------------------------------------
# Explicitly drop personal and non-ML fields
# ---------------------------------------------------------------------
DROP_COLUMNS = [
    "name",
    "email",
    "review",
    "folder_name",
    "anonymize",
    "canceled",
]

df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

# ---------------------------------------------------------------------
# Export anonymized ML input
# ---------------------------------------------------------------------
df.to_csv("exports/c_min_k_max_params.csv", index=False)

print("Exported anonymized file: ./exports/c_min_k_max_params.csv")
print("Personal identifiers and free-text fields were removed.")
