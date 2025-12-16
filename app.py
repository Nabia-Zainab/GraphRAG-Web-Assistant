import streamlit as st
import os
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from graph_builder import GraphBuilder
from rag_chain import GraphRAGChain
from web_loader import WebLoader
from streamlit_agraph import agraph, Node, Edge, Config
from dotenv import load_dotenv

load_dotenv(override=True)

st.set_page_config(page_title="GraphRAG Web Scraper", layout="wide")

st.title("🕸️ GraphRAG Web: Chat with the Internet")
st.markdown("Powered by **Groq (Llama 3.1)**, **Neo4j** & **BeautifulSoup**")

# Tabs
tab1, tab2 = st.tabs(["💬 Chat", "🕸️ Graph Visualization"])

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = GraphRAGChain()

if "graph_builder" not in st.session_state:
    st.session_state.graph_builder = GraphBuilder()

# Sidebar for URL Input
with st.sidebar:
    st.header("🌐 Website Input")
    url_input = st.text_input("Enter Website URL", placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence")
    
    # Advanced Options for Deep Scraping
    with st.expander("⚙️ Advanced Scraping Options"):
        max_depth = st.slider("Depth (How deep to click links)", 1, 3, 2)
        max_pages = st.number_input("Max Pages to Scrape", min_value=1, max_value=50, value=10)
    
    if st.button("Process Website") and url_input:
        with st.spinner(f"Crawling {url_input} (Max {max_pages} pages)... This may take a while."):
            try:
                # 1. Load Content (Recursive)
                loader = WebLoader(url_input, max_depth=max_depth, max_pages=max_pages)
                docs = loader.load()
                
                if not docs:
                    st.error("Failed to load content. Please check the link or try a different one.")
                else:
                    st.success(f"Successfully scraped {len(docs)} pages!")
                    
                    # 2. Split Text (Reduced size to avoid LLM overload)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                    splits = text_splitter.split_documents(docs)
                    st.info(f"Split into {len(splits)} chunks (Chunk Size: 500).")
                    
                    # 3. Ingest into Graph
                    st.text(f"Building Knowledge Graph from {len(splits)} chunks...")
                    st.session_state.graph_builder.ingest_documents(splits)
                    
                    # 4. Add to Vector Store
                    st.text("Updating Vector Index...")
                    st.session_state.rag_chain.add_documents_to_vector_store(splits)
                    
                    st.success("Website Processed! You can now chat with its content.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Chat Interface
with tab1:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the website..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking (Traversing Graph + Searching Vectors)..."):
                try:
                    chain = st.session_state.rag_chain.get_chain()
                    response = chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Graph Visualization
with tab2:
    st.header("Knowledge Graph View")
    st.markdown("Interative view: You can zoom, drag nodes, and hover to see details.")
    
    if st.button("Refresh Graph"):
        # Hum RAG chain se graph instance pass karenge
        if 'graph_builder' in st.session_state:
             from visualizer import visualize_graph
             visualize_graph(st.session_state.graph_builder.graph)
        else:
             st.warning("Please process a website first.")
