import pandas as pd
import json
import pandas as pd
import json
import sys
from sapsam import parser, constants
import sys
import io
from collections import defaultdict
from typing import Dict
import igraph as ig
import networkx as nx
import iplotx as ipx
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import pandas as pd
import ollama
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pydantic import BaseModel, TypeAdapter
from typing import List

def filter_bpmn_models(csv_path, min_elements=10, max_elements=50):
    high_quality_list = []
    
    # Process in chunks to manage memory (SAP-SAM files are large)
    for chunk in pd.read_csv(csv_path, chunksize=5000):
        # 1. Filter by Namespace (BPMN 2.0 only)
        bpmn_only = chunk[chunk['Namespace'].str.contains('bpmn2.0', na=False)]
        
        for _, row in bpmn_only.iterrows():
            try:
                model_json = json.loads(row['Model JSON'])
                # Count elements in the root of the model
                num_elements = len(model_json.get('childShapes', []))
                
                # 2. Filter by element count and ensure it has a valid name
                if min_elements <= num_elements <= max_elements and pd.notnull(row['Name']):
                    high_quality_list.append({
                        'Model ID': row['Model ID'],
                        'Name': row['Name'],
                        'Element Count': num_elements,
                        'Model JSON': row['Model JSON']
                    })
            except (json.JSONDecodeError, TypeError):
                continue
    
    return pd.DataFrame(high_quality_list)

# Usage


def json_to_labeled_graph(model_json_str):
    model_data = json.loads(model_json_str)
    G = nx.DiGraph()
    
    # 1. First pass: Collect all shapes into a dictionary for quick lookup
    all_shapes = {}
    def collect(shapes):
        for s in shapes:
            all_shapes[s["resourceId"]] = s
            if "childShapes" in s: collect(s["childShapes"])
    collect(model_data.get("childShapes", []))

    # 2. Second pass: Build Nodes and Labeled Edges
    def build(shapes):
        for shape in shapes:
            res_id = shape.get("resourceId")
            stencil = shape.get("stencil", {}).get("id")
            name = shape.get("properties", {}).get("name", "")

            # Only add non-flow elements as actual Nodes
            if stencil != "SequenceFlow":
                # Node Label: [Type] Name
                node_label = f"[{stencil}] {name}" if name else stencil
                G.add_node(res_id, label=node_label, type=stencil)

                # Find Outgoing connections
                for out in shape.get("outgoing", []):
                    out_id = out.get("resourceId")
                    out_shape = all_shapes.get(out_id)

                    # If outgoing is a SequenceFlow, get ITS label and target
                    if out_shape and out_shape.get("stencil", {}).get("id") == "SequenceFlow":
                        edge_label = out_shape.get("properties", {}).get("name", "")
                        targets = out_shape.get("outgoing", [])
                        if targets:
                            G.add_edge(res_id, targets[0]["resourceId"], label=edge_label)
                    else:
                        G.add_edge(res_id, out_id, label="")

            if "childShapes" in shape:
                build(shape["childShapes"])

    build(model_data.get("childShapes", []))
    return G



def json_to_detailed_graph(model_json_str):
    model_data = json.loads(model_json_str)
    G = nx.DiGraph()
    
    # helper to clean stencil IDs (e.g., "UserTask" -> "User Task")
    def format_type(s_id):
        import re
        return re.sub(r"(\w)([A-Z])", r"\1 \2", s_id)

    def process_shapes(shapes):
        for shape in shapes:
            res_id = shape.get("resourceId")
            stencil = shape.get("stencil", {}).get("id")
            name = shape.get("properties", {}).get("name", "Untitled")
            
            # 1. Identify Task specific details
            # Signavio IDs are specific: 'UserTask', 'ServiceTask', 'CollapsedSubprocess'
            task_execution_type = format_type(stencil)
            
            # 2. Construct the detailed label
            # Format: "Task Name (Type: User Task)"
            detailed_label = f"{name}\n({task_execution_type})"
            
            if stencil != "SequenceFlow":
                G.add_node(res_id, 
                           label=detailed_label, 
                           task_type=stencil,
                           original_name=name)

                # 3. Handle Connections (Edges)
                for out in shape.get("outgoing", []):
                    G.add_edge(res_id, out.get("resourceId"))

            # Recurse for nested elements
            if "childShapes" in shape:
                process_shapes(shape["childShapes"])

    process_shapes(model_data.get("childShapes", []))
    return G




# Example output: "Approve Invoice (Type: User Task)"



def generate_summary(node_data,edge_data):
    prompt = f"""          
          Consider the following process data represented as a graph. Nodes represent task or function and
          an edge represent flow between two tasks where task in the `from' node should be executed before task in `to' node. Outcome of task execution 
          at the `from' node will be passed to the task in the `to' node.  
          The data has two dataframes. In node_data there are four fields, (a) "id" (it means an unique id for the node), 
          (b)"label" (it means an description of the node), 
          (c)"task_type" (it means the task that the node executes), and (d)"original_name" 
          (means an alternate name of it or description of the task or function executed by the node).
          The edge_data contains three fields as (a)from_node, (b)to_node, and (c)label(may describe why these 
          two nodes are connected).  

          Based on this process, generate a scenario by creating a description of what each node performs and a description
          on why two nodes are connected.  
        
          Combine all descriptions into one single paragraph. Only output this single paragraph.

          node_data: \"{node_data}\"
          edge_data: \"{edge_data}\"
          
    """
    response = ollama.generate(model="llama3.2", prompt=prompt, stream=False)
    generated_text = response['response']
    return generated_text



def get_process_data():

    csv_paths = parser.get_csv_paths()
    csv_path = csv_paths[0]

    df_high_quality = filter_bpmn_models(csv_path)
    graph = json_to_labeled_graph(df_high_quality.iloc[1]['Model JSON'])
    node_data = graph.nodes.data()
    edge_data = graph.edges.data()
    G_1 = json_to_detailed_graph(df_high_quality.iloc[1]['Model JSON'])

    #print('G_1',G_1.nodes.data())
    #print('G_1',G_1.edges.data())

    node_data_csv = []
    edge_data_csv = []

    id  = []
    label = []
    task_type = []
    original_name = []

    n_data = G_1.nodes.data()

    for d in n_data:
        if len(d[1])>0:
            id.append(d[0])
            label.append(d[1]['label'])
            task_type.append(d[1]['task_type'])
            original_name.append(d[1]['original_name'])
        else:
            id.append('')
            label.append('')
            task_type.append('')
            original_name.append('')
            
    node_data = {"id":id,"label":label,"task_type":task_type,"original_name":original_name}
    node_df = pd.DataFrame(node_data)
    #print('node df',node_df)

    edge_data = graph.edges.data()
    #print('edge_data',edge_data)
    from_node = []
    to_node = []
    label = []
    for edge in edge_data:
        from_node.append(edge[0])
        to_node.append(edge[1])
        if len(edge[2]['label'])>0:
            label.append(edge[2]['label'])
        else:
            label.append('')

    edge_data = {"from_node":from_node,"to_node":to_node,"label":label}
    edge_df = pd.DataFrame(edge_data)
    #print('edge_df',edge_df)



    process_summary = generate_summary(node_df,edge_df)
    return(process_summary)

