'''
class for turning each region of an image into a node, computing spatial relationships by bbox and using it as attributes
for arcs.
So given a set of images the output will be a scene graph for each img.
'''

import json
import os
from pathlib import Path
import networkx as nx
from spatial_relations import *
import numpy as np
import torch
from torch_geometric.data import Data


RELATION_TO_IDX = {"above": 0, "left": 1, "right": 2, "overlaps": 3, "adjacent" : 4}

class GraphScene:


    def __init__(self, json_folder, city ):
        self.json_folder = json_folder
        self.city = city



    def region_reader_from_json(self, file_path):
        '''

        Args:
            file: json file with image's metadata

        Returns:
            a list of region where for each region contain all metadata (class_id, class_name and bbox)

        '''
        regions = []
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
            for region in data["regions"]:
                regions.append(region)
        return regions



    def nodes_maker(self, regions, file, embedding_dict):
        '''

        Args:
            regions: list of regions with metadata
            file: name of the json file

        Returns:
            G: a graph that contain one node for each region

        '''
        G = nx.Graph()
        G.graph['name'] = file
        base_name = os.path.splitext(file)[0] #taking json file name without extension.
        for idx, region in enumerate(regions):
            #key needed to find embedding for that image in that region
            key = f"{self.city}_{base_name}_region_{idx}"
            embedding = embedding_dict.get(key)
            if embedding is not None:
                #idx is used as identifier for regions.
                G.add_node(idx, **region)
                G.nodes[idx]["embedding"] = embedding
        if file:
            #associate the json name file as name for the graph.
            G.graph['name'] = file
        return G

    def arcs_maker(self, G):
        '''

        Args:
            G: graph with no arcs, only nodes

        Returns:
            G: the graph with arcs based on space relation computed using bboxes.

        '''
        for i in G.nodes():
            for j in G.nodes():
                if i == j:
                    continue
                bbox1 = G.nodes[i]["bbox"]
                bbox2 = G.nodes[j]["bbox"]
                class_a = G.nodes[i]["class_name"]
                class_b = G.nodes[j]["class_name"]

                #binary vector in which each element represent a possible spatial relation
                vector = [0] * len(RELATION_TO_IDX)
                if is_above(bbox1, bbox2):
                    vector[RELATION_TO_IDX["above"]] = 1
                if is_left_of(bbox1, bbox2):
                    vector[RELATION_TO_IDX["left"]] = 1
                if overlaps(bbox1, bbox2, class_a, class_b):
                    vector[RELATION_TO_IDX["overlaps"]] = 1
                if is_adjacent(bbox1, bbox2):
                    vector[RELATION_TO_IDX["adjacent"]] = 1
                #it there is at leat a one
                if any(vector):
                    G.add_edge(i, j, relation = vector)
        return G

    def graph_maker(self):
        '''
        for each file in the json directory create a scene graph
        Returns:

        '''
        G_list = []
        G = nx.DiGraph()
        embedding_dict = np.load(f"data/processed/{self.city}/{self.city}_clip.npy", allow_pickle=True).item()
        for file in os.listdir(self.json_folder):
            regions = self.region_reader_from_json(os.path.join(self.json_folder, file))
            G = self.nodes_maker(regions, file, embedding_dict)
            G = self.arcs_maker(G)
            G_list.append(G)

            '''DESCRIPTION OF THE CREATED GRAPH'''
            print(f"Descrizione del scene graph relativo a {file}")
            for node_id, attrs in G.nodes(data=True):
                print(f"Nodo {node_id}: class_name = {attrs['class_name']}")
                print(f"  Embedding (prime 5 dim): {attrs['embedding'][:5]}")

            '''
            idx_to_relation = {v: k for k, v in RELATION_TO_IDX.items()}

            for u, v, attrs in G.edges(data=True):
                rel_vec = attrs["relation"]
                rel_names = [idx_to_relation[i] for i, val in enumerate(rel_vec) if val == 1]
                print(f" {u} → {v} ({G.nodes[u]['class_name']} → {G.nodes[v]['class_name']}): {', '.join(rel_names)}")
            '''
        return G_list

    #Data is the pytorch geometric format for representing graph
    @staticmethod
    def convert_nx_to_pyg(graph_nx: nx.DiGraph) -> Data:
        # Estrai embedding come feature dei nodi
        node_feats = [graph_nx.nodes[n]["embedding"] for n in graph_nx.nodes()]
        #convert in tensors
        x = torch.tensor(node_feats, dtype=torch.float)

        # Estrai indici degli archi
        edge_index = []
        edge_attr = []
        for u, v, data in graph_nx.edges(data=True):
            edge_index.append([u, v])
            edge_attr.append(data["relation"])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Opzionale: salva class_name nel dizionario del nodo (solo il nome della classe)
        class_names = [graph_nx.nodes[n]["class_name"] for n in graph_nx.nodes()]

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, class_name=class_names)

    @staticmethod
    def save_pyg_graphs(graphs, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for i, data in enumerate(graphs):
            torch.save(data, os.path.join(save_dir, f"graph_{i}.pt"))


print("📂 Working directory:", os.getcwd())
os.chdir(Path(__file__).resolve().parent.parent.parent)
print("📂 Working directory:", os.getcwd())

cities = ["Bari", "Shibuya"]
for city in cities:
    path = f"data/processed/{city}/masked_output/labels/"
    graph = GraphScene(path, city)
    G_list = graph.graph_maker()
    # Converti in PyG
    pyg_graphs = [GraphScene.convert_nx_to_pyg(g) for g in G_list]
    # Salva
    GraphScene.save_pyg_graphs(pyg_graphs, save_dir=f"data/gnn_ready_graphs/{graph.city}")
