import csv
import math
import os

# Define paths
data_dir = "../../Data"
output_dir = "."

mimic_path = os.path.join(data_dir, "mimic_septic_shock_tabular.csv")
ami_path = os.path.join(data_dir, "cleaned.mi (myocardial infarction).csv")

def get_correlations(file_path, output_csv):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)

    # Find numeric columns that aren't mostly missing
    num_rows = len(data)
    numeric_indices = []
    col_names = []
    
    for i, col in enumerate(header):
        vals = []
        missing_count = 0
        for row in data:
            if i < len(row) and row[i].strip() != "":
                try:
                    vals.append(float(row[i]))
                except ValueError:
                    break
            else:
                missing_count += 1
        
        if len(vals) > num_rows * 0.05: # At least 5% data present
            if len(vals) == (num_rows - missing_count): # All non-empty were float
                numeric_indices.append(i)
                col_names.append(col)

    # Calculate correlation matrix
    matrix = [[1.0 for _ in range(len(numeric_indices))] for _ in range(len(numeric_indices))]
    
    # Pre-calculate data for each column
    col_data = []
    for idx in numeric_indices:
        vals = []
        for row in data:
            if idx < len(row) and row[idx].strip() != "":
                try:
                    vals.append(float(row[idx]))
                except ValueError:
                    vals.append(None)
            else:
                vals.append(None)
        col_data.append(vals)

    for i in range(len(numeric_indices)):
        for j in range(i + 1, len(numeric_indices)):
            # Pairwise complete observations
            xi_list = []
            xj_list = []
            for k in range(num_rows):
                if col_data[i][k] is not None and col_data[j][k] is not None:
                    xi_list.append(col_data[i][k])
                    xj_list.append(col_data[j][k])
            
            if len(xi_list) < 2:
                r = 0
            else:
                mean_i = sum(xi_list) / len(xi_list)
                mean_j = sum(xj_list) / len(xj_list)
                
                num = sum((xi - mean_i) * (xj - mean_j) for xi, xj in zip(xi_list, xj_list))
                den_i = sum((xi - mean_i) ** 2 for xi in xi_list)
                den_j = sum((xj - mean_j) ** 2 for xj in xj_list)
                
                if den_i == 0 or den_j == 0:
                    r = 0
                else:
                    r = num / math.sqrt(den_i * den_j)
            
            matrix[i][j] = r
            matrix[j][i] = r

    # Write output
    with open(os.path.join(output_dir, output_csv), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([""] + col_names)
        for i, row in enumerate(matrix):
            writer.writerow([col_names[i]] + [f"{v:.4f}" for v in row])

    print(f"Saved {output_csv}")

# Run
get_correlations(mimic_path, "correlation_matrix_mimic.csv")
get_correlations(ami_path, "correlation_matrix_ami.csv")
