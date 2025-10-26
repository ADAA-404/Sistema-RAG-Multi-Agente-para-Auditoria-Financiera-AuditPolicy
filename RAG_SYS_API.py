#!/usr/bin/env python
# coding: utf-8

# In[3]:


# RAG_SYS_API
import uvicorn
import os
import sys
import pickle
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

import json



# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- IMPORTACIONES DEL SISTEMA RAG ---
from RAG_SYS import CustomRetriever, AuditPolicyRAGWorkflow, process_files_recursively, split_documents

# CONSTANTES GLOBALES
PERSIST_DIR = "./chroma_db"
VECTOR_COLLECTION = "audit_policies"
CHUNK_PICKLE_FILE = "persisted_chunks.pkl"
DOCUMENTS_DIRECTORY = "Data Files"

# Variable global para el retriever y la app
app = FastAPI(title="AuditPolicy RAG Service")
global_retriever = None # Se inicializará al inicio del servidor
global_workflow = None

# ----------------------------------------------------
# Pydantic Schemas para la API
# ----------------------------------------------------
class QuestionRequest(BaseModel):
    """Esquema para la solicitud de preguntas"""
    question: str
    
class AnswerResponse(BaseModel):
    """Esquema para la respuesta de la API"""
    question: str
    answer: str
    relevance_check: str
    verification_report: Dict


# ----------------------------------------------------
# 1. Función de Inicialización (Startup Event)
# ----------------------------------------------------

# Esta función ejecuta la lógica de ingesta/carga una sola vez al iniciar el servidor
@app.on_event("startup")
async def startup_event():
    """
    Inicializa el sistema RAG, incluyendo la carga o creación de ChromaDB
    y la inicialización del Hybrid Retriever.
    """
    global global_retriever, global_workflow

    logger.info("⚡ Iniciando el servicio RAG de Auditoría...")
    
    chunks = []
    
    # Lógica de persistencia adaptada para el servidor (puede cambiar segun lo que se use)
    ingest_required = not (os.path.exists(PERSIST_DIR) and os.path.exists(os.path.join(PERSIST_DIR, 'chroma.sqlite3')))

    if ingest_required:
        logger.info(" Base de datos NO detectada. Ejecutando Ingesta completa (Lenta)...")
        try:
            # 1. Lectura y Chunking (Lento)
            documents = process_files_recursively(DOCUMENTS_DIRECTORY)
            if not documents:
                 logger.error("No se encontraron documentos para procesar.")
                 raise RuntimeError("No hay documentos para iniciar el servicio.")
                 
            chunks = split_documents(documents)
            logger.info(f"Documentos fragmentados en {len(chunks)} chunks.")
            
            # 2. Persistir los chunks
            with open(CHUNK_PICKLE_FILE, "wb") as f:
                pickle.dump(chunks, f)
            logger.info(f" Chunks serializados y guardados.")
            
        except Exception as e:
            logger.error(f"Error fatal durante la ingesta: {e}")
            raise RuntimeError(f"Fallo al cargar documentos: {e}")

    else:
        logger.info(f" Base de datos existente detectada. Intentando cargar chunks para BM25...")
        try:
            # 1. Intentar cargar chunks (Rápido)
            with open(CHUNK_PICKLE_FILE, "rb") as f:
                chunks = pickle.load(f)
            logger.info(f" {len(chunks)} chunks cargados.")
            
        except FileNotFoundError:
            # Si ChromaDB existe pero el .pkl se perdió, re-leemos y re-chunkamos (Tolerable)
            logger.warning("Archivo persisted_chunks.pkl NO encontrado. Re-leyendo y Re-fragmentando documentos (Tolerable)...")
            documents = process_files_recursively(DOCUMENTS_DIRECTORY) # <-- LECTURA LENTA AQUÍ
            chunks = split_documents(documents)
            logger.info(f"Documentos re-fragmentados en {len(chunks)} chunks. BM25 actualizado.")
            with open(CHUNK_PICKLE_FILE, "wb") as f:
                pickle.dump(chunks, f)
        except Exception as e:
            logger.error(f"Fallo al cargar chunks o re-leer documentos: {e}")
            raise RuntimeError("Fallo en la persistencia de chunks.")


    # 2. Inicializar el CustomRetriever (Carga ChromaDB o lo crea si hizo ingesta)
    try:
        global_retriever = CustomRetriever(chunks)
        global_workflow = AuditPolicyRAGWorkflow(global_retriever)
        logger.info(" Servicio RAG inicializado exitosamente.")
    except Exception as e:
        logger.error(f"Error al inicializar CustomRetriever/Workflow: {e}")
        raise RuntimeError("Fallo al inicializar componentes RAG.")


# ----------------------------------------------------
# 2. Endpoint Principal: /ask
# ----------------------------------------------------

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Procesa una pregunta utilizando el flujo RAG inicializado.
    """
    global global_workflow
    
    if global_workflow is None:
        # Esto solo debería ocurrir si el startup event falló
        raise HTTPException(status_code=503, detail="El sistema RAG aún no está inicializado. Intente más tarde.")

    question = request.question
    logger.info(f"-> Recibida pregunta: {question}")
    
    try:
        # Ejecutar el workflow completo
        result = global_workflow.full_pipeline(question)
        
        # El resultado incluye la respuesta, el chequeo de relevancia y la verificación
        
        # Intentar parsear el JSON de verificación para el output
        try:
             verification_data = json.loads(result['verification_report'])
        except:
             # Si no es un JSON válido, enviar el texto crudo
             verification_data = {"error": "Failed to parse JSON", "raw_report": result['verification_report']}

        return AnswerResponse(
            question=question,
            answer=result.get("draft_answer", "No se pudo generar respuesta."),
            relevance_check=result.get("relevance_check", "N/A"),
            verification_report=verification_data
        )
    
    except Exception as e:
        logger.error(f"Error durante el procesamiento de la pregunta: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del RAG: {str(e)}")


# In[ ]:




