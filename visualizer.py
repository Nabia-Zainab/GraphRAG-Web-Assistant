from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components
import os

def visualize_graph(graph_instance):
    # 1. Initialize PyVis Network
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # Physics settings - Nodes ko door rakhne k liye
    net.force_atlas_2based()
    
    # 2. Define Colors for different Node Types (Groups)
    # Aap apni pasand k colors yahan set kr skti hain
    color_map = {
        "Organization": "#ff5733",  # Red-Orange (Brands/Stores)
        "Brand": "#ff5733",         # Red-Orange
        "Product": "#3380ff",       # Blue
        "Material": "#28a745",      # Green
        "Color": "#ffc107",         # Yellow
        "Feature": "#6c757d",       # Gray
        "Location": "#6f42c1"       # Purple
    }
    
    # Default color agar koi type match na ho
    default_color = "#97c2fc"

    # 3. Get Data from Neo4j (Limit 100 to avoid crash)
    query = """
    MATCH (n)-[r]->(m)
    RETURN n.id AS source, labels(n) AS source_labels, 
           type(r) AS rel_type, 
           m.id AS target, labels(m) AS target_labels
    LIMIT 70
    """
    
    try:
        results = graph_instance.query(query)
    except Exception as e:
        st.error(f"Graph query failed: {e}")
        return

    # 4. Add Nodes and Edges to PyVis
    for record in results:
        source = record['source']
        target = record['target']
        rel_type = record['rel_type']
        
        # Determine Source Group & Color
        s_labels = record['source_labels']
        s_group = s_labels[0] if s_labels else "Unknown"
        s_color = color_map.get(s_group, default_color)
        
        # Determine Target Group & Color
        t_labels = record['target_labels']
        t_group = t_labels[0] if t_labels else "Unknown"
        t_color = color_map.get(t_group, default_color)

        # Add Nodes (Size bada karein agar wo Brand/Org hai)
        s_size = 25 if s_group in ["Organization", "Brand"] else 15
        t_size = 25 if t_group in ["Organization", "Brand"] else 15

        net.add_node(source, label=source, title=f"Type: {s_group}", color=s_color, size=s_size)
        net.add_node(target, label=target, title=f"Type: {t_group}", color=t_color, size=t_size)
        
        # Add Edge (Arrow)
        net.add_edge(source, target, title=rel_type, label=rel_type, color="#d3d3d3")

    # 5. Physics Options (Is se graph thoda smooth move karega)
    net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 16,
          "face": "tahoma"
        }
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": false
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)

    # 6. Save and Read HTML
    try:
        path = "graph_viz.html"
        net.save_graph(path)
        
        # Read file to render in Streamlit
        with open(path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        components.html(html_content, height=600, scrolling=True)
        
    except Exception as e:
        st.error(f"Visualization Error: {e}")
