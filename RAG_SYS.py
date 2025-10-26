#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Imprime la versión de PyTorch
print(f"PyTorch Version: {torch.__version__}")

# Verifica si CUDA (GPU) está disponible [ahorra mucho tiempo]
if torch.cuda.is_available():
    print(f"CUDA está disponible: ¡Sí! 🟢")
    print(f"Nombre de la GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    print(f"   Allocated: {torch.cuda.memory_allocated(0)/1e9:.4f} GB")
else:
    print("CUDA está disponible: No 🔴. Usando CPU.")

import numpy as np
print(np.__version__)
print("NumPy loaded successfully!")


# In[2]:


# ===============================================
# BLOQUE DE LIBRERÍAS NECESARIAS
# ===============================================

from typing import Dict, List, TypedDict, Optional, Union
import time
import os
import pickle
import json
import logging

# Configuración básica de logging para ver mensajes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Framework de Agentes ---
from langgraph.graph import StateGraph, END  

# --- LangChain Core y Componentes ---
from langchain_core.documents import Document                
from langchain_ollama import ChatOllama     
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma          

# --- Componentes de Retrieval ---
from langchain_community.retrievers import BM25Retriever     # Este sí funciona
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter 
# -----------------------------------------------

from docling.document_converter import DocumentConverter

from typing import List

from langchain_community.chat_models import ChatOllama

from langchain_community.document_loaders import PyPDFLoader # Nueva librería

os.environ["OLLAMA_KEEP_ALIVE"] = "30m"


# In[3]:


# Definición de funciones de utilidad (Global)

def process_files_recursively(documents_dir: str) -> List[Document]:
    """Lee recursivamente todos los PDFs y retorna una lista plana de Documentos de LangChain."""
    
    logger.info(f"Iniciando lectura recursiva de documentos desde: {documents_dir}")
    all_documents = []

    for root, _, files in os.walk(documents_dir):
        for filename in files:
            if filename.endswith(".pdf"):
                filepath = os.path.join(root, filename)
                logger.info(f"Procesando archivo: {filepath}")
                try:
                    # Usar el cargador estándar de LangChain
                    loader = PyPDFLoader(filepath)
                    pages = loader.load()
                    all_documents.extend(pages) # Añadir las páginas como Documentos individuales
                except Exception as e:
                    logger.error(f"Error procesando {filepath} con PyPDFLoader: {e}")
    
    logger.info(f"Extracción de texto completada desde {len(all_documents)} páginas/documentos.")
    return all_documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Toma una lista de Documentos y los fragmenta en chunks más pequeños."""
    
    # Concatenamos todo el contenido en un solo string para el chunking global (Lógica actual funcional)
    all_text = "\n\n".join([doc.page_content for doc in documents])
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n", ".", " ", ""],
        chunk_size=1500, 
        chunk_overlap=150,
    )
    chunks = text_splitter.create_documents([all_text])
    
    return chunks


# In[4]:


PERSIST_DIR = "./chroma_db"
VECTOR_COLLECTION = "audit_policies"

class CustomRetriever:
    """
    Construye y combina la recuperación Vectorial (Chroma) y BM25 (Keyword)
    para simular el EnsembleRetriever sin usar la clase obsoleta de LangChain.
    """
    def __init__(self, chunks: List[Document], embed_model_name: str = "all-MiniLM-L6-v2"):
        self.chunks = chunks
        
        # 1. Configuración de Embeddings - Implementacion de CUDA
        # Usamos 'cuda:0' para la RTX (Caso personal)
        model_kwargs = {'device': 'cuda:0'} 
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model_name,
            model_kwargs=model_kwargs 
        )
        
        # 2. Construir VectorStore y Retriever (Semántico)
        if os.path.exists(PERSIST_DIR) and os.path.exists(os.path.join(PERSIST_DIR, 'chroma.sqlite3')):
            # Si existe, cargar desde el disco (Evita sobrecarga)
            logger.info(f" Cargando ChromaDB desde el disco: {PERSIST_DIR}. Saltando ingesta.")
            self.vectorstore = Chroma(
                collection_name=VECTOR_COLLECTION,
                embedding_function=self.embeddings,
                persist_directory=PERSIST_DIR
            )
        else:
            # Si NO existe, construir e inicializar (Solo la primera vez se cargan los documentos)
            logger.info(f" Ingestando 77 documentos y construyendo nueva ChromaDB en {PERSIST_DIR}...")
            if not os.path.exists(PERSIST_DIR):
                os.makedirs(PERSIST_DIR)
                
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=VECTOR_COLLECTION,
                persist_directory=PERSIST_DIR,
            )
            # Guardar explícitamente los datos para futuras cargas
            self.vectorstore.persist() 
            logger.info(" Ingesta completa. Base de datos guardada para futuros usos.")

        # 2.2. Configuración del Retriever (Semántico y Keyword)
        # Asumiendo un ajuste moderado el k = 12
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 12}) 
        
        # El BM25 debe construirse con los 'chunks' en memoria, no es persistente por defecto
        # Asumiendo que funciona ya el BM25 aquí:
        from langchain_community.retrievers import BM25Retriever
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
        self.bm25_retriever.k = 12 #  USAR k=12 para consistencia y calidad


    def hybrid_retrieve(self, query: str, k: int = 12) -> List[Document]:
        """Ejecuta la búsqueda híbrida combinando los resultados."""
        
        # Obtener los mejores resultados de cada retriever
        vector_results = self.vector_retriever.invoke(query)
        bm25_results = self.bm25_retriever.invoke(query)
        
        # Combinar resultados
        combined_results = vector_results + bm25_results
        
        # Desduplicar y priorizar (Mantenemos el orden de aparición)
        unique_contents = set()
        final_docs = []
        for doc in combined_results:
            if doc.page_content not in unique_contents:
                unique_contents.add(doc.page_content)
                final_docs.append(doc)
                
        # Devolver los top K documentos
        return final_docs[:k]


# In[5]:


# Inicializar el modelo base una sola vez
# Asegurarze de que Ollama esté corriendo en http://localhost:11434 en paralelo a las pruebas o ejecución
BASE_LLM = ChatOllama(model="llama3.1:8b", temperature=0.0)


# In[6]:


# Modificación Prompts de RelevanceChecker
class RelevanceChecker:
    def __init__(self, llm=BASE_LLM):
        # Usamos el modelo base inicializado
        self.llm = llm 

    def check(self, question: str, retriever: 'CustomRetriever', k=12) -> str:
        
        # 1. Recuperar documentos relevantes usando el CustomRetriever
        docs = retriever.hybrid_retrieve(question, k=k)
        context = "\n".join([doc.page_content for doc in docs])

        # Si la pregunta contiene cualquiera de estos términos clave, es RELEVANTE.
        KEYWORDS_TO_FORCE_RELEVANCE = [
            "niif", "nic", "sox", "ifric", "arrendamiento", 
            "auditoría", "contabilidad", "cumplimiento", 
            "control interno", "financiero"
        ]

        # Convierte la pregunta a minúsculas para una coincidencia insensible a mayúsculas
        q_lower = question.lower()
        
        # Verifica si alguna de las palabras clave existe en la pregunta
        if any(keyword in q_lower for keyword in KEYWORDS_TO_FORCE_RELEVANCE):
            return "RELEVANTE" # ¡Salida directa, sin llamar al LLM!

        # Si llega aquí, la pregunta es genérica o irrelevante. Usamos el LLM para decidir.
        # 2. Definición del Prompt (ajustes para su optimización)
        prompt = f"""
        Eres un motor de clasificación **BINARIA** de preguntas para un sistema RAG de Auditoría. y Finanzas.
        Tu única respuesta debe ser una de estas dos palabras, en mayúsculas: 'RELEVANTE' o 'NO RELEVANTE'.

        PREGUNTA: "{question}"

        CONTEXTO RECUPERADO:
        ---
        {context}
        ---

       Instrucciones de Reglas:
        1. RELEVANTE: La respuesta es 'RELEVANTE' **SOLO SI** la PREGUNTA contiene **explícitamente** una de las siguientes palabras clave (no hay inferencia):
            - NIIF
            - NIC
            - SOX
            - IFRIC
            - Contabilidad
            - Auditoría
            - Financiero
            - Arrendamiento

        2. NO RELEVANTE: Si la pregunta **no tiene relación** con los conceptos anteriores (Ej de relacion con: animales, historia, cultura, deportes, o trivilidades).
        """
        
        # 3. Llamada al LLM (Sustitución de self.model.chat por self.llm.invoke)
            
        try:
            response = self.llm.invoke(prompt)
            llm_response = response.content.strip().upper()

            if llm_response == 'RELEVANTE':
                return 'RELEVANTE'
            else:
                # Si el modelo divaga o dice algo diferente, lo tratamos como no relevante
                return 'NO RELEVANTE'
        except Exception as e:
            # Manejo de error del LLM
            print(f"Error en RelevanceChecker: {e}")
            return 'ERROR'


# In[7]:


# Modificación de ResearchAgent (Ajustes de Prompt)
class ResearchAgent:
    def __init__(self, llm=BASE_LLM):
        # Se puede usar una temperatura un poco más alta para la creatividad y síntesis
        self.llm = ChatOllama(model="llama3.1:8b", temperature=0.3)

    def invoke(self, state: Dict) -> Dict:
        # 1. Extraer los datos del diccionario de estado
        question = state["question"]
        documents = state["documents"]
        
        # 2. Reutilizar la lógica existente de generación
        context = "\n---\n".join([f"Fragmento {i+1}: {doc.page_content}" for i, doc in enumerate(documents)])
        
        # Prompt de generación
        prompt = f"""
        Eres un Consultor Senior de PwC para Auditoría y Contabilidad, experto en NIIF, SOX y mas temas de LEYES.
        Tu misión es generar un borrador de respuesta preciso, conciso y profesional 
        para la pregunta de auditoría basada ÚNICAMENTE en el contexto proporcionado.

        PREGUNTA: "{question}"

        CONTEXTO DE NORMAS:
        ---
        {context}
        ---

        Instrucciones CRÍTICAS:
        1. **Tu única TAREA es responder al texto de la consulta USANDO EXCLUSIVAMENTE el CONTEXTO.**
        2. **NO se permite el rechazo:** NO uses frases como "La pregunta no está claramente formulada", "parece estar ausente", o "no hay pregunta específica".
        3. **Si la información es insuficiente:** Synthetize lo mejor posible y añade una **Nota al final** indicando que la respuesta es incompleta o requiere más contexto.
        4. Tu respuesta debe ser una **síntesis directa** de la información relevante.
        """

        # Llamada al LLM
        response = self.llm.invoke(prompt)
        draft_answer = response.content.strip()
        
        # 3. Retornar el estado actualizado (solo la respuesta generada)
        return {"draft_answer": draft_answer} # LangGraph se encarga de fusionar esto con el estado existente


# In[8]:


# Modificación de VerificationAgent (Ajuste sobre el prompt)
class VerificationAgent:
    def __init__(self, llm=BASE_LLM):
        # Establer la temperatura en 0.01 para ser estricto y no creativo
        self.llm = ChatOllama(model="llama3.1:8b", temperature=0.01)

    def check(self, state: Dict) -> Dict:
            
        # Extraer los datos del diccionario de estado
        question = state["question"]
        draft_answer = state["draft_answer"]
        documents = state["documents"]
        
        # Reutilizar la lógica existente
        context = "\n---\n".join([doc.page_content for doc in documents])

        # Se activa si el ResearchAgent devolvió un mensaje de rechazo genérico
        rejection_keywords = ["no hay suficiente información", "no tiene relación", "no está presente"]
        draft_answer = state.get("draft_answer", "")
        
        if any(keyword in draft_answer.lower() for keyword in rejection_keywords):
            import json # Importar si no está al inicio del archivo (o falla en orden de ejecucion)
            # Devolver un JSON predefinido para garantizar el formato y la justificación (Detalle de salida)
            return {
            "fidelidad": "INACEPTABLE",
            "justificacion": "La respuesta fue un rechazo profesional y no aborda la pregunta. La fidelidad al objetivo es INACEPTABLE.",
            "hallazgo_critico": False
            }

        # Prompt de verificación
        prompt = f"""
        Eres un Revisor Senior de Calidad de Auditoría estricto de la firma PwC.
        Tu tarea es verificar la exactitud y fidelidad de la 'RESPUESTA BORRADOR' frente al 'CONTEXTO DE NORMAS'[cite: 31, 32].
        Tu única salida debe ser un JSON **estricto y válido**, sin texto de introducción ni conclusión.
        
        PREGUNTA ORIGINAL: "{question}"
        RESPUESTA BORRADOR: "{draft_answer}"
        
        CONTEXTO DE NORMAS:
        ---
        {context}
        ---
        
        Instrucciones:
        1. **Evalúa la FIDELIDAD y el ABORDAJE:** ¿La RESPUESTA BORRADOR **aborda directamente** el tema principal de la PREGUNTA y está soportada por el CONTEXTO DE NORMAS?
        2. **CASTIGO (Fidelidad BAJA/INACEPTABLE):** Si la RESPUESTA BORRADOR contiene frases de evasión (como "La pregunta parece faltar", "No hay pregunta específica" o "Intento responder a posibles preguntas"), la fidelidad es **INACEPTABLE**.
        3. **CASTIGO (Alucinación):** Si la RESPUESTA BORRADOR introduce hechos NO vistos en el CONTEXTO, es **INACEPTABLE**.
        
        El formato es un JSON estricto y único:
        ```json
        {{
          "fidelidad": "ALTA" | "MEDIA" | "BAJA" | "INACEPTABLE",
          "justificacion": "Explicación breve de la calificación (Ej: La respuesta no aborda la pregunta o alucina, o está perfectamente respaldada).",
          "hallazgo_critico": false
        }}
        """

        # Llamada al LLM
        response = self.llm.invoke(prompt)
        verification_report = response.content.strip()
        
        return {"verification_report": verification_report}


# In[9]:


# ORQUESTACIÓN LANGGRAPH (El Flujo de Trabajo princpial)
class GraphState(TypedDict):
    """Representa el estado del grafo de agentes."""
    question: str
    draft_answer: Optional[str]
    documents: List[Document]
    verification_report: Optional[str]
    relevance_check: str

class AuditPolicyRAGWorkflow:
    def __init__(self, retriever: CustomRetriever):
        self.retriever = retriever
        self.relevance_checker = RelevanceChecker()
        self.research_agent = ResearchAgent()
        self.verification_agent = VerificationAgent()
        self.workflow = self._build_workflow()
    
    def retrieve_documents(self, state: GraphState) -> Dict:
        """Paso inicial de recuperación de documentos."""
        question = state["question"]
        docs = self.retriever.hybrid_retrieve(question)
        logger.info(f"Documentos recuperados: {len(docs)}")
        return {"documents": docs}

    def check_relevance_node(self, state: GraphState) -> Dict:
        """Nodo que llama al Agente de Relevancia."""
        question = state["question"]
        # Se usa el retriever directo en el agente, pero el estado pasa la pregunta
        check_result = self.relevance_checker.check(question, self.retriever) 
        return {"relevance_check": check_result}

    def decide_to_research(self, state: GraphState) -> str:
        """Función de enrutamiento que decide si investigar o finalizar."""
        if state["relevance_check"] == 'RELEVANTE':
            return 'research'
        else:
            return 'end_no_research'

    def finalize_no_research(self, state: GraphState) -> Dict:
        """
        Nodo de limpieza para preguntas NO RELEVANTES. 
        Asegura que el estado final contenga todas las claves esperadas por la API.
        """
        # Inyectamos un informe de verificación formal de "N/A"
        return {
            "draft_answer": "N/A",
            "verification_report": json.dumps({
                "fidelidad": "NO APLICA",
                "justificacion": "La pregunta fue clasificada como NO RELEVANTE para el dominio de Auditoría/Finanzas y el proceso terminó.",
                "hallazgo_critico": False
            })
        }

    def _build_workflow(self):
        """Construye y compila el grafo de LangGraph."""
        workflow = StateGraph(GraphState)

        # 1. Recuperar documentos
        workflow.add_node("retrieve", self.retrieve_documents)
        
        # 2. Verificar Relevancia (si vale la pena continuar)
        workflow.add_node("check_relevance", self.check_relevance_node)
        
        # 3. Generar respuesta
        workflow.add_node("research", self.research_agent.invoke)
        
        # 4. Verificar respuesta (Control de Calidad)
        workflow.add_node("verify", self.verification_agent.check)

        # 5. NUEVO NODO DE LIMPIEZA
        workflow.add_node("finalize_no_research", self.finalize_no_research)

        # Definir el flujo de control
        workflow.set_entry_point("retrieve")
        
        # Flujo 1: Documentos -> Verificar Relevancia
        workflow.add_edge("retrieve", "check_relevance")

        # Flujo 2: Enrutamiento (Decisión)
        workflow.add_conditional_edges(
            "check_relevance",
            self.decide_to_research,
            {'research': 'research', 'end_no_research': 'finalize_no_research'}
        )

        # Flujo 3: Investigación -> Verificación
        workflow.add_edge("research", "verify")

        workflow.add_edge("finalize_no_research", END) # Nuevo flujo de Limpieza

        # Flujo 4: Verificación -> Final
        workflow.add_edge("verify", END)

        return workflow.compile()
    
    def full_pipeline(self, question: str) -> Dict:
        """Ejecuta el pipeline completo y retorna el resultado."""
        return self.workflow.invoke({"question": question})


# In[10]:


def process_documents_and_answer_questions(documents_dir: str, questions: List[str]):
    # --- 1. Preparación de Datos (Docling y Chunking) ---
    logger.info(f"Iniciando procesamiento de documentos de forma recursiva desde: {documents_dir}")
    all_text = ""
    processed_files = 0
    found_files = []
    
    # Usamos os.walk para recorrer recursivamente todas las subcarpetas
    for root, _, files in os.walk(documents_dir):
        for filename in files:
            if filename.endswith(".pdf"):
                filepath = os.path.join(root, filename)
                logger.info(f"Procesando archivo: {filepath}")
                try:
                    # Usar el cargador estándar de LangChain, que funciona con casi todos los PDFs.
                    loader = PyPDFLoader(filepath)
                    # Cargamos y dividimos en documentos (Langchain Document objects)
                    pages = loader.load()
    
                    # Añadimos los documentos cargados a la lista de chunks
                    for page in pages:
                        all_text += page.page_content + "\n\n"
                    
                    processed_files += 1
                    found_files.append(filepath)
                except Exception as e:
                    logger.error(f"Error procesando {filepath} con PyPDFLoader: {e}")
    
    if processed_files == 0:
        logger.error("No se encontró ningún archivo PDF válido para procesar.")
        return []

    logger.info(f"Extracción de texto completada desde {processed_files} documentos.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n", ".", " ", ""],
        chunk_size=1500, 
        chunk_overlap=150,
    )
    chunks = text_splitter.create_documents([all_text])
    logger.info(f"Documentos fragmentados en {len(chunks)} chunks.")

    # --- 2. Construcción del Retriever Híbrido ---
    retriever = CustomRetriever(chunks)
    logger.info("Retriever Híbrido (Chroma+BM25) construido.")
    
    # --- 3. Ejecución del Workflow ---
    workflow = AuditPolicyRAGWorkflow(retriever)
    results = []
    
    for i, question in enumerate(questions):
        logger.info(f"\n===== PROCESANDO PREGUNTA {i+1}/{len(questions)}: {question} =====")
        try:
            result = workflow.full_pipeline(question)
            
            # Formatear el resultado para la salida
            final_result = {
                "question": question,
                "answer": result.get("draft_answer", "N/A"),
                "verification": result.get("verification_report", "N/A"),
                "relevance_check": result.get("relevance_check", "N/A")
            }
            results.append(final_result)
            
        except Exception as e:
            logger.error(f"Error al ejecutar el workflow para la pregunta: {e}")
            results.append({"question": question, "answer": f"Error: {str(e)}", "verification": "Error", "relevance_check": "ERROR"})
            
    return results


# In[11]:


# La función de ejecución debe cambiar su nombre y lógica para reflejar que solo procesa preguntas.

def run_pipeline_for_questions(retriever: CustomRetriever, questions: List[str]) -> List[Dict]:
    """Ejecuta el workflow RAG para una lista de preguntas, usando un retriever ya inicializado."""
    
    # La ingesta y chunking se hace en el 'main'.
    
    # Construcción del Workflow (usa el retriever que le pasamos anteriormente)
    workflow = AuditPolicyRAGWorkflow(retriever)
    results = []
   
    for i, question in enumerate(questions):
        logger.info(f"\n===== PROCESANDO PREGUNTA {i+1}/{len(questions)}: {question} =====")
        try:
            result = workflow.full_pipeline(question)
            
            # Formatear el resultado para la salida
            final_result = {
                "question": question,
                "answer": result.get("draft_answer", "N/A"),
                "verification": result.get("verification_report", "N/A"),
                "relevance_check": result.get("relevance_check", "N/A")
            }
            results.append(final_result)
            
        except Exception as e:
            logger.error(f"Error al ejecutar el workflow para la pregunta: {e}")
            results.append({"question": question, "answer": f"Error: {str(e)}", "verification": "Error", "relevance_check": "ERROR"})
            
    return results


# In[12]:


if __name__ == "__main__":
    # Se debe crear una carpeta 'data' y colocar los PDFs (NIIF, SOX, etc.) dentro o formatear de donde se consulta la informacion.
    PERSIST_DIR = "./chroma_db"
    VECTOR_COLLECTION = "audit_policies"
    CHUNK_PICKLE_FILE = "persisted_chunks.pkl" # <-- NECESARIO para pickle
    DOCUMENTS_DIRECTORY = "Data Files" # <-- Definición de ruta
    
    # Preguntas de prueba que exigen razonamiento y cumplimiento normativo
    test_questions = [
        "¿Cuáles son los requisitos clave para la divulgación de información financiera relacionada con la sostenibilidad, según las NIIF?",
        "¿Qué secciones de la Ley Sarbanes-Oxley (SOX) son más relevantes para el control interno de las PYMES?",
        "Resuma las implicaciones contables de un contrato de arrendamiento a 5 años, según la ifric16.",
        "¿Cuál es el color favorito de un panda?" # Pregunta irrelevante para probar el RelevanceChecker
    ]

    chunks = []
    
    # Crea la carpeta de datos si no existe
    if not os.path.exists(DOCUMENTS_DIRECTORY):
        os.makedirs(DOCUMENTS_DIRECTORY)
        logger.warning(f"Carpeta '{DOCUMENTS_DIRECTORY}' creada. Coloque sus PDFs aquí.")
        # Si la carpeta es nueva, forzamos la ingesta completa
        ingest_required = True

    # Condición de Ingesta: Si ChromaDB NO existe.
    ingest_required = not (os.path.exists(PERSIST_DIR) and os.path.exists(os.path.join(PERSIST_DIR, 'chroma.sqlite3')))
    

    if ingest_required:
        # LÓGICA DE INGESTA COMPLETA (LENTA) - Solo si ChromaDB no existe.
        logger.info(" Base de datos NO detectada. Ejecutando Ingesta completa (Lenta)...")
        
        # 1. Lectura y Chunking (Lento)
        documents = process_files_recursively(DOCUMENTS_DIRECTORY)
        if not documents:
             logger.error("No se encontraron documentos para procesar.")
             sys.exit(1)
             
        chunks = split_documents(documents)
        logger.info(f"Documentos fragmentados en {len(chunks)} chunks.")
        
        # 2. Persistir los chunks (Para la próxima vez que se inicie y Chroma ya exista; se ahorra tiempo)
        try:
            with open(CHUNK_PICKLE_FILE, "wb") as f:
                pickle.dump(chunks, f)
            logger.info(f" Chunks serializados y guardados en {CHUNK_PICKLE_FILE}.")
        except Exception as e:
            logger.error(f"Advertencia: No se pudo crear {CHUNK_PICKLE_FILE}: {e}")

    else:
        # LÓGICA DE CARGA RÁPIDA - ChromaDB existe.
        logger.info(f" Base de datos existente detectada. Intentando cargar chunks para BM25...")
        
        # 1. Intentar cargar chunks (Rápido)
        try:
            with open(CHUNK_PICKLE_FILE, "rb") as f:
                chunks = pickle.load(f)
            logger.info(f" {len(chunks)} chunks cargados desde {CHUNK_PICKLE_FILE}.")
            
        except FileNotFoundError:
            # Si ChromaDB existe pero el .pkl se perdió, re-leemos y re-chunkamos (Tolerable)
            logger.warning("Archivo persisted_chunks.pkl NO encontrado. Re-leyendo y Re-fragmentando documentos (Tolerable)...")
            documents = process_files_recursively(DOCUMENTS_DIRECTORY) # <-- LECTURA LENTA AQUÍ
            chunks = split_documents(documents)
            logger.info(f"Documentos re-fragmentados en {len(chunks)} chunks. BM25 actualizado.")
            # Volver a guardar el PKL
            with open(CHUNK_PICKLE_FILE, "wb") as f:
                pickle.dump(chunks, f)

    retriever = CustomRetriever(chunks)
        
    # Ejecutar el pipeline
    final_results = run_pipeline_for_questions(retriever, test_questions)
    
    # Imprimir resultados finales
    print("\n" + "="*50)
    print("           RESULTADOS FINALES DEL AUDITPOLICY-RAG")
    print("="*50)
    
    for i, result in enumerate(final_results):
        print(f"\n--- PREGUNTA {i+1} ---")
        print(f"Q: {result['question']}")
        print(f"Relevancia detectada: {result['relevance_check']}")
        print(f"Respuesta generada: \n{result['answer']}")
        
        try:
            # Intenta imprimir el JSON de verificación de forma bonita
            verification_data = json.loads(result['verification'])
            print(f"Informe de Verificación (Fidelidad): {verification_data['fidelidad']}")
            print(f"Justificación: {verification_data['justificacion']}")
        except:
            # Si no es JSON (ej. si el check falló o terminó antes)
            print(f"Informe de Verificación: {result['verification']}")


# In[ ]:





# In[ ]:




