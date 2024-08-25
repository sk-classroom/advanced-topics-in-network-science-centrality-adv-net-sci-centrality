# %% Import
from scipy import sparse
import pandas as pd
import igraph
import numpy as np

node_table = pd.read_csv(
    "https://raw.githubusercontent.com/skojaku/adv-net-sci-course/main/data/airport_network_v2/node_table.csv"
)
edge_table = pd.read_csv(
    "https://raw.githubusercontent.com/skojaku/adv-net-sci-course/main/data/airport_network_v2/edge_table.csv"
)
src, trg = tuple(edge_table[["src", "trg"]].values.T)
edge_list = tuple(zip(src, trg))

# node_id and name dictionary
n_nodes = node_table.shape[0]
id2name = np.array([""] * n_nodes, dtype="<U64")
id2name[node_table["node_id"]] = node_table["Name"].values

g = igraph.Graph(
    edge_list,
    vertex_attrs=dict(Name=id2name, node_id=node_table["node_id"].values),
)


# %% Test -----------

frac_nodes_removed, connectivity = robustness_index_random(g, n_nodes_removed = 10)

target_robustness_index = 0.35
robustness_index = np.mean(connectivity)
assert robustness_index > target_robustness_index, f"""
❌ The robustness profile of random attacks does not match the expected answer. \n
  - Expected Minimum Robustness Index: {target_robustness_index}. \n
  - Actual Robustness Index: {robustness_index}. \n
The actual robustness index must be greater than the expected minimum robustness index.
"""
print("✅ The robustness profile of random attacks matches the expected answer.")


frac_nodes_removed, connectivity = robustness_index_degree(g, n_nodes_removed = 10)
target_robustness_index = 0.078950
robustness_index = np.mean(connectivity)
assert np.abs(robustness_index - target_robustness_index) < 0.01, f"""
❌ The robustness profile of degree attacks does not match the expected answer. \n
  - Expected Robustness Index: {target_robustness_index}. \n
  - Actual Robustness Index: {robustness_index}. \n
The actual robustness index must be within 0.01 of the expected robustness index.
"""
print("✅ The robustness profile of degree attacks matches the expected answer.")

frac_nodes_removed, connectivity = robustness_index_closeness(g, n_nodes_removed = 10)
target_robustness_index = 0.378030
robustness_index = np.mean(connectivity)
assert np.abs(robustness_index - target_robustness_index) < 0.01, f"""
❌ The robustness profile of closeness-based attacks does not match the expected answer. \n
  - Expected Robustness Index: {target_robustness_index}. \n
  - Actual Robustness Index: {robustness_index}. \n
The actual robustness index must be within 0.01 of the expected robustness index.
"""
print("✅ The robustness profile of closeness-based attacks matches the expected answer.")

frac_nodes_removed, connectivity = robustness_index_betweenness(g, n_nodes_removed = 10)

target_robustness_index = 0.0469
robustness_index = np.mean(connectivity)
assert np.abs(robustness_index - target_robustness_index) < 0.01, f"""
❌ The robustness profile of betweenness-based attacks does not match the expected answer. \n
  - Expected Robustness Index: {target_robustness_index}. \n
  - Actual Robustness Index: {robustness_index}. \n
The actual robustness index must be within 0.01 of the expected robustness index.
"""
print("✅ The robustness profile of betweenness-based attacks matches the expected answer.")