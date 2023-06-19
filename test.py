import networkx as nx
import community as community_louvain

str_adj_list = '0 1 2 3 4 13 20 34 42 77 106,1,2 15 25 31,3 5 11 17 27 65 67 91 110,4 64 66 85 90 114 248,5 6 9 10 12 22 28 30 35 36 38 39 45 51 58 71 72 99 101 115 128 154 155 169 208 231,6 7 8 14 18 19 48 50 59 61 87 122 140 168 170 182 198 199,7,8,9 43 44 96,10 52 137 144,11 46 111 192,12,13 16 24 29 56 69 80 93 95 136 149 157 224,14,15 21 23 68 70 75 76 88 102 141 159 235 237,16 74 134 161,17 183,18 55 81 82 84 86 209 244,19,20 32 173,21 26 121 152 179,22,23,24 107,25 33 100 249,26,27,28,29 229,30,31 49 54,32 60 104 139 212 227,33,34 37,35 73 92 202,36,37 57 127,38,39 40,40 41 228,41 222 233,42,43,44 47 53 62 78 160 164 188 232,45 63 166 175 190 220,46 153,47,48 120 132 150 223,49 226 234,50,51 79 191,52 83,53 89,54 109 138 151 176 189 214 216 225,55,56 116 215,57,58 211,59,60,61 126,62,63,64 197,65 98,66,67 163,68,69,70 194,71 184,72 219,73 147,74 167 186,75,76 94 130 217,77,78,79,80 123 124 156 200 218,81,82,83,84 133,85,86,87 135,88 97 143,89,90 119,91 185,92,93,94 113 178 207,95,96 105 108 145 146 177 245 247,97 174,98,99 112 203 243,100 103 193,101,102,103,104 129,105,106,107 172,108 241,109,110 125 162,111 118 242,112 165,113,114,115 117,116,117,118,119,120 131 246,121 240,122,123 148 204,124,125,126,127,128,129,130,131 187,132,133,134,135,136,137,138,139,140,141 142 180,142,143,144,145,146,147,148 181,149,150,151,152,153 171,154,155 205,156 158,157,158,159,160,161,162,163 221,164 195 210,165,166,167,168,169,170,171,172 196,173,174 206,175,176,177 201,178,179,180,181,182,183,184,185 236,186,187,188 230,189,190,191,192,193,194,195,196,197,198,199,200 213,201,202,203,204,205,206,207,208,209,210,211,212 238,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237 239,238,239,240,241,242,243,244,245,246,247,248,249'
split_adj_list = str_adj_list.split(',')
entries = [split_str for split_str in split_adj_list]
nodes = [int(entry.split()[0]) for entry in entries]
print(nodes)

def get_community_node_dict(partition):
    """
    Get dictionaries mapping community IDs to node IDs and vice versa.

    Args:
        partition (dict): A dictionary with node IDs as keys and community IDs as values.

    Returns:
        tuple: A tuple containing two dictionaries:
            - node_community_dict (dict): A dictionary mapping node IDs to community IDs.
            - community_node_dict (dict): A dictionary mapping community IDs to lists of node IDs.
    """
    # create a dictionary with node ids as keys and community ids as values
    node_community_dict = {int(node): int(community_id) for node, community_id in partition.items()}
    print(node_community_dict)
    # this dict contains the community ids as keys and the node ids as values
    community_node_dict = {}
    for node, community_id in node_community_dict.items():
        if community_id not in community_node_dict:
            community_node_dict[community_id] = [node]
        else:
            community_node_dict[community_id].append(node)
    print(community_node_dict)
    return node_community_dict, community_node_dict

graph = nx.parse_adjlist(split_adj_list)
# apply louvain method on the graph
partition = community_louvain.best_partition(graph)
print(partition)
node_community_dict, community_node_dict = get_community_node_dict(partition)

for node in nodes:
    community_id = node_community_dict[node]
    community_nodes = community_node_dict[community_id]
