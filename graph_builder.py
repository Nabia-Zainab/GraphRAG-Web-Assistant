import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(override=True)

# --- Define Data Models (FIXED) ---
class Node(BaseModel):
    id: str = Field(description="Unique identifier for the entity. MUST be the full, specific name (e.g., 'Dr. Sarah Jenkins' instead of 'Sarah').")
    type: str = Field(description="Type of the entity (e.g., Person, Organization, Concept)")

class Relationship(BaseModel):
    source: str = Field(description="Source entity ID")
    target: Optional[str] = Field(default=None, description="Target entity ID")
    type: str = Field(description="Type of relationship (e.g., WORKS_FOR, LOCATED_IN)")
    description: Optional[str] = Field(default=None, description="Brief description of the relationship context")

class GraphData(BaseModel):
    nodes: List[Node]
    relationships: List[Relationship]

# --- Graph Builder Logic ---
class GraphBuilder:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        # Try-Except block to handle different LangChain versions for Neo4j
        try:
            from langchain_neo4j import Neo4jGraph
            self.graph = Neo4jGraph(
                url=os.getenv("NEO4J_URI"),
                username=os.getenv("NEO4J_USERNAME"),
                password=os.getenv("NEO4J_PASSWORD")
            )
        except ImportError:
            self.graph = Neo4jGraph(
                url=os.getenv("NEO4J_URI"),
                username=os.getenv("NEO4J_USERNAME"),
                password=os.getenv("NEO4J_PASSWORD")
            )

    def extract_graph_data(self, text: str) -> GraphData:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledge graph extractor. Extract entities (nodes) and relationships from the text.
            
            CRITICAL GUIDELINES:
            1. NODES: Identify key entities (Person, Organization, Product, Concept).
            2. RELATIONSHIPS: MUST have a 'source', 'target', and 'type'.
               - ❌ BAD:  {{ "source": "Apple", "type": "GROWS_ON" }}
               - ✅ GOOD: {{ "source": "Apple", "target": "Tree", "type": "GROWS_ON" }}
            3. ATTRIBUTES: If it's a property like Price or Color, link it to the Product.
               - ✅ GOOD: {{ "source": "Shirt", "target": "Blue", "type": "HAS_COLOR" }}
            4. IDS: Use the exact literal string for IDs to ensure merging.
            
            Return JSON matching the schema exactly."""),
            ("human", "Text: {text}")
        ])
        
        # Use structured output with the fixed model
        chain = prompt | self.llm.with_structured_output(GraphData)
        return chain.invoke({"text": text})

    def ingest_documents(self, documents: List[Document]):
        print(f"Processing {len(documents)} documents...")
        
        # Optional: Clear database before starting (Uncomment if needed)
        # self.graph.query("MATCH (n) DETACH DELETE n")
        
        for i, doc in enumerate(documents):
            try:
                print(f"Analyzing chunk {i+1}/{len(documents)}...")
                data = self.extract_graph_data(doc.page_content)
                
                # Create Nodes
                if data.nodes:
                    for node in data.nodes:
                        clean_id = node.id.strip() 
                        self.graph.query(
                            "MERGE (n:Entity {id: $id}) SET n.type = $type",
                            {"id": clean_id, "type": node.type}
                        )
                
                # Create Relationships
                if data.relationships:
                    for rel in data.relationships:
                        if not rel.target:
                            continue
                            
                        clean_source = rel.source.strip()
                        clean_target = rel.target.strip()
                        # Default description if None
                        desc = rel.description if rel.description else ""
                        
                        cypher = f"""
                        MATCH (s:Entity {{id: $source}})
                        MATCH (t:Entity {{id: $target}})
                        MERGE (s)-[r:{rel.type.upper().replace(' ', '_')}]->(t)
                        SET r.description = $description
                        """
                        self.graph.query(
                            cypher,
                            {
                                "source": clean_source, 
                                "target": clean_target, 
                                "description": desc
                            }
                        )
                print(f"✔ Chunk {i+1} ingested successfully.")
                
            except Exception as e:
                # Error print karega lekin process nahi rokega
                print(f"⚠ Error processing chunk {i+1}: {e}")

        print("Graph ingestion complete.")