"""
routers/documents.py - GET /documents and DELETE /documents/{id} endpoints.
Place this file at: rag-chatbot/routers/documents.py
"""
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import vectorstore

router = APIRouter()


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    ingested_at: str = ""
    chunk_count: int = 0
    total_sections: int = 0


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


class DeleteResponse(BaseModel):
    success: bool
    message: str
    doc_id: str


@router.get("/documents", response_model=DocumentListResponse, summary="List all ingested documents")
async def list_documents():
    """Returns a list of all documents that have been ingested into the RAG system."""
    docs = vectorstore.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentInfo(
                doc_id=d.get("doc_id", ""),
                filename=d.get("filename", "unknown"),
                ingested_at=d.get("ingested_at", ""),
                chunk_count=d.get("chunk_count", 0),
                total_sections=d.get("total_sections", 0),
            )
            for d in docs
        ],
        total=len(docs),
    )


@router.delete("/documents/{doc_id}", response_model=DeleteResponse, summary="Delete a document")
async def delete_document(doc_id: str):
    """
    Delete a document and all its chunks from the vector store.
    Note: hnswlib doesn't support hard vector deletion.
    Deleted document chunks are removed from metadata and won't appear in search results.
    """
    success = await vectorstore.delete_document(doc_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not found"
        )

    return DeleteResponse(
        success=True,
        message=f"Document '{doc_id}' and all its chunks have been deleted.",
        doc_id=doc_id,
    )
