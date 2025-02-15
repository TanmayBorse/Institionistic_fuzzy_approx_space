import pandas as pd

data = {
    'Institution': ['Inst_1', 'Inst_2', 'Inst_3', 'Inst_4', 'Inst_5', 'Inst_6', 'Inst_7', 'Inst_8', 'Inst_9', 'Inst_10'],
    'IC': [229, 227, 226, 191, 179, 148, 131, 124, 88, 92],
    'IF': [151, 143, 145, 110, 117, 102, 78, 61, 58, 48],
    'PP': [304, 298, 266, 316, 247, 180, 138, 130, 121, 100],
    'RS': [169, 169, 167, 163, 160, 147, 145, 142, 143, 137],
    'SS': [56, 53, 54, 41, 53, 43, 46, 38, 40, 32],
    'ECA': [49, 79, 63, 64, 53, 27, 25, 9, 34, 2]
}

# Range for each feature
range_dict = {
    "IC": [1, 250],
    "IF": [1, 200],
    "PP": [1, 385],
    "RS": [1, 200],
    "SS": [1, 60],
    "ECA": [1, 80]
}

# Define Weights for Features (User Input)
weights = {
    "IC": 0.2,
    "IF": 0.005,
    "PP": 0.45,
    "RS": 0.2,
    "SS": 0.1,
    "ECA": 0.0007
}

df = pd.DataFrame(data)
df

def calculate_membership(Vix, Viy, max_value):
    return round(1 - abs(Vix - Viy) / max_value, 3)

def fuzzy_grouping_by_membership(df, range_dict, alpha=0.92):
    feature_groups = {}
    membership_values = {}

    for feature in df.columns[1:]:  # Skip 'Institution'
        max_value = range_dict[feature][1]
        memberships = []

        # Calculate membership values
        for i in range(len(df)):
            row_membership = [calculate_membership(df[feature][i], df[feature][j], max_value) for j in range(len(df))]
            memberships.append(row_membership)

        # Store membership values as DataFrame
        membership_values[feature] = pd.DataFrame(memberships, columns=df['Institution'], index=df['Institution'])

        # Start with individual clusters
        clusters = [[df['Institution'][i]] for i in range(len(df))]

        # Merge clusters based on fuzzy proximity relation μR(x, y) ≥ α
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if memberships[i][j] >= alpha:
                    cluster_i = next(cluster for cluster in clusters if df['Institution'][i] in cluster)
                    cluster_j = next(cluster for cluster in clusters if df['Institution'][j] in cluster)

                    if cluster_i != cluster_j:  
                        cluster_i.extend(cluster_j)
                        clusters.remove(cluster_j)

        feature_groups[feature] = clusters

    return feature_groups, membership_values


fuzzy_groups, membership_values = fuzzy_grouping_by_membership(df, range_dict)

def compute_Z_and_C(df, range_dict, fuzzy_groups, weights):
    Z_values = {inst: [] for inst in df['Institution']}
    C_values = {inst: 0 for inst in df['Institution']}
    weighted_C_values = {inst: 0 for inst in df['Institution']}

    for feature, clusters in fuzzy_groups.items():
        max_value = range_dict[feature][1]

        # Compute Z values
        for cluster in clusters:
            m = len(cluster)
            sum_values = sum(df[df['Institution'] == inst][feature].values[0] for inst in cluster)
            Z_value = round(sum_values / (m * max_value), 3)

            for inst in cluster:
                Z_values[inst].append(Z_value)

    # Compute C values and weighted c value
    for inst in df['Institution']:
        C_values[inst] = round(sum(Z_values[inst]), 3)
        weighted_C_values[inst] = round(sum(Z_values[inst][i] * list(weights.values())[i] for i in range(len(weights))), 3)

    return Z_values, C_values, weighted_C_values

Z_values, C_values, weighted_C_values = compute_Z_and_C(df, range_dict, fuzzy_groups, weights)

Z_table = pd.DataFrame(Z_values, index=df.columns[1:]).transpose()
Z_table['C_value'] = Z_table.sum(axis=1)

# Prepare the weighted final table
Weighted_Z_table = Z_table.copy()
Weighted_Z_table['Weighted_C_value'] = list(weighted_C_values.values())

# Round values to 2 decimal places
Z_table = Z_table.round(2)
Weighted_Z_table = Weighted_Z_table.round(3)

# Determine the best institution(s) based on C value
max_C_value = Z_table['C_value'].max()
best_institutions = Z_table[Z_table['C_value'] == max_C_value].index.tolist()

# Determine the best institution(s) based on Weighted C value
max_weighted_C_value = Weighted_Z_table['Weighted_C_value'].max()
best_weighted_institutions = Weighted_Z_table[Weighted_Z_table['Weighted_C_value'] == max_weighted_C_value].index.tolist()

for feature, membership_df in membership_values.items():
    print(f"\nMembership Values for Feature {feature}:")
    print(membership_df.round(3))


for feature, clusters in fuzzy_groups.items():
    print(f"\nFuzzy Groups for Feature {feature}:")
    for cluster in clusters:
        print(cluster)


import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
for feature, membership_df in membership_values.items():
    print(f"\nMembership Values for Feature {feature}:")
    membership_df['Institution'] = membership_df.index
    parallel_coordinates(membership_df, 'Institution', color=plt.cm.Set1.colors)
    plt.title(f'Parallel Coordinates Plot for Membership Values')
    plt.show()


print(Z_table)

print(Weighted_Z_table)

print(f"\n\nThe best institution based on C value: {', '.join(best_institutions)} with C value = {max_C_value}")
print(f"\nThe best institution based on Weighted C value: {', '.join(best_weighted_institutions)} with Weighted C value = {max_weighted_C_value}\n")
