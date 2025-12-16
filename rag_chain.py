import os
from typing import List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv(override=True)

class GraphRAGChain:
    def __init__(self):
        # 1. Initialize LLM
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # 2. Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 3. Initialize Vector Store (Chroma)
        self.vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )
        
        # 4. Initialize Graph Store (Neo4j)
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )

    def add_documents_to_vector_store(self, documents: List[Document]):
        self.vector_store.add_documents(documents)
        
    def get_graph_context(self, query: str) -> str:
        # Step A: Extract Entity
        prompt = ChatPromptTemplate.from_template(
            """Extract the single most important entity (Product, Brand, or Category) from this query.
            Return ONLY the name.
            Query: {query}
            Entity:"""
        )
        chain = prompt | self.llm | StrOutputParser()
        entity = chain.invoke({"query": query}).strip()
        print(f"🔍 Graph Looking for: {entity}")

        # Step B: Robust Graph Traversal (Fetch Properties too)
        # Hum node ki properties (price, description) bhi return karwayenge
        cypher = """
        MATCH (start:Entity)
        WHERE toLower(start.id) CONTAINS toLower($entity)
        MATCH path = (start)-[r*1..2]-(connected)
        UNWIND relationships(path) AS rel
        RETURN 
            startNode(rel).id AS source, 
            type(rel) AS rel_type, 
            endNode(rel).id AS target,
            properties(endNode(rel)) AS target_props
        LIMIT 100
        """
        
        try:
            result = self.graph.query(cypher, {"entity": entity})
            
            context = []
            for record in result:
                # Basic Relationship
                fact = f"{record['source']} {record['rel_type']} {record['target']}"
                
                # Check for Properties (Price, etc.)
                props = record['target_props']
                if props:
                    # Filter out internal ID properties
                    clean_props = {k: v for k, v in props.items() if k not in ['id', 'type']}
                    if clean_props:
                        fact += f" (Details: {clean_props})"
                
                context.append(fact)
            
            return "\n".join(list(set(context)))
            
        except Exception as e:
            print(f"Graph Error: {e}")
            return ""

    def get_chain(self):
        # 1. Vector Retriever (Increased k to 6 for more info)
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 6})
        
        def hybrid_retrieval(query):
            # A. Vector Context
            vector_docs = vector_retriever.invoke(query)
            vector_context = "\n".join([d.page_content for d in vector_docs])
            
            # B. Graph Context
            graph_context = self.get_graph_context(query)
            
            print(f"📄 Graph Context Found (Size: {len(graph_context)} chars)")
            
            return f"""
            --- WEBSITE CONTENT ---
            {vector_context}
            
            --- DATABASE RELATIONSHIPS ---
            {graph_context}
            """
            
        # 2. Final Answer Prompt
        template = """You are a helpful Shopping Assistant for Brandmarkt.
        Use the provided Information to answer the customer's question.
        
        Rules:
        1. If you find a specific price, mention it clearly.
        2. If the user asks about a Category (e.g., Bags/Taschen) and you don't see specific items, 
           list the BRANDS available in that category instead.
        3. Be polite and professional.
        4. If info is completely missing, suggest they visit the store in Winterthur.
        
        Information:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": hybrid_retrieval, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain