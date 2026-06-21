import hashlib
import logging
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger("haystack_diagnostics")


def _stream_document_batches(
    document_store,
    batch_size: int = 1000,
    include_embeddings: bool = False,
    filters: Optional[Dict[str, Any]] = None,
) -> Iterator[List[Any]]:
    """
    Streams documents in batches from the Document Store to prevent OOM errors in production.
    Supports:
    - InMemoryDocumentStore
    - WeaviateDocumentStore (via collection.iterator)
    - QdrantDocumentStore (via client.scroll)
    - Fallback: standard filter_documents (with warning)
    """
    store_class = document_store.__class__.__name__

    # 1. InMemoryDocumentStore
    if store_class == "InMemoryDocumentStore":
        try:
            documents = document_store.filter_documents(filters=filters)
        except Exception:
            try:
                documents = document_store.filter_documents()
            except Exception:
                documents = document_store.filter_documents(filters=None)
        
        for i in range(0, len(documents), batch_size):
            yield documents[i : i + batch_size]

    # 2. WeaviateDocumentStore
    elif store_class == "WeaviateDocumentStore":
        try:
            collection = document_store.collection
            properties = [p.name for p in collection.config.get().properties]
            
            # Using Weaviate v4 collection.iterator()
            iterator = collection.iterator(
                include_vector=include_embeddings,
                return_properties=properties
            )
            
            batch = []
            for doc in iterator:
                batch.append(document_store._to_document(doc))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        except Exception as e:
            logger.warning(f"Error lazy-streaming from Weaviate: {str(e)}. Falling back to standard filter_documents.")
            try:
                documents = document_store.filter_documents(filters=filters)
            except Exception:
                documents = document_store.filter_documents()
            for i in range(0, len(documents), batch_size):
                yield documents[i : i + batch_size]

    # 3. QdrantDocumentStore
    elif store_class == "QdrantDocumentStore":
        try:
            client = document_store.client
            collection_name = document_store.index
            
            offset = None
            while True:
                scroll_result, next_offset = client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    with_payload=True,
                    with_vectors=include_embeddings,
                    offset=offset
                )
                
                if not scroll_result:
                    break
                
                batch = []
                for point in scroll_result:
                    if hasattr(document_store, "_to_document"):
                        batch.append(document_store._to_document(point))
                    else:
                        from haystack.dataclasses import Document
                        payload = point.payload or {}
                        batch.append(Document(
                            id=point.id,
                            content=payload.get("content"),
                            meta=payload.get("meta", {}),
                            embedding=point.vector if include_embeddings else None
                        ))
                
                if batch:
                    yield batch
                
                if next_offset is None or len(scroll_result) < batch_size:
                    break
                offset = next_offset
        except Exception as e:
            logger.warning(f"Error lazy-streaming from Qdrant: {str(e)}. Falling back to standard filter_documents.")
            try:
                documents = document_store.filter_documents(filters=filters)
            except Exception:
                documents = document_store.filter_documents()
            for i in range(0, len(documents), batch_size):
                yield documents[i : i + batch_size]

    # 4. Fallback for other stores
    else:
        logger.warning(
            f"Document store of type '{store_class}' does not support paginated streaming. "
            "Falling back to full filter_documents scan, which may consume significant memory."
        )
        try:
            documents = document_store.filter_documents(filters=filters)
        except Exception:
            try:
                documents = document_store.filter_documents()
            except Exception:
                documents = document_store.filter_documents(filters=None)
        
        for i in range(0, len(documents), batch_size):
            yield documents[i : i + batch_size]


def validate_document_store(
    document_store,
    expected_metadata_keys: Optional[List[str]] = None,
    expected_embedding_dim: Optional[int] = None,
    short_chunk_threshold: int = 50,
    filters: Optional[Dict[str, Any]] = None,
    batch_size: int = 1000,
    validate_embeddings: bool = True,
) -> Dict[str, Any]:
    """
    Validates the health of the documents within a Haystack Document Store.
    Performs 7 ingestion-layer checks in a streaming batch-wise manner:
    1. content is None
    2. Empty documents
    3. Very short chunks
    4. Duplicate chunks (MD5 hash)
    5. Missing metadata fields
    6. Null embeddings
    7. Embedding dimension mismatch

    NOTE: Duplicate detection requires maintaining content hash indexes in memory.
    This scales linearly with the number of unique documents (O(N) memory complexity).

    :param document_store: The Haystack DocumentStore instance.
    :param expected_metadata_keys: Optional list of keys that must be present in document metadata.
    :param expected_embedding_dim: Optional dimension expected for embeddings.
    :param short_chunk_threshold: Character threshold for identifying short chunks.
    :param filters: Optional filters dictionary to restrict which documents are validated.
    :param batch_size: Size of batches to stream from database.
    :param validate_embeddings: Whether to fetch and validate embeddings (turn off to optimize memory/speed).
    :return: A dictionary containing the validation summary and detailed results.
    """
    total_docs = 0
    
    # Trackers for issues
    content_none_ids = []
    empty_content_ids = []
    short_chunk_ids = []
    null_embedding_ids = []
    
    # Hash map for duplicate detection (O(N) memory on hashes and IDs)
    hash_map = defaultdict(list)
    
    # Metadata tracker
    missing_metadata_details = []
    
    # Embedding dimensions tracker
    dim_map = defaultdict(list)
    dim_mismatch_details = []
    
    invalid_doc_ids = set()

    # Stream the documents in batches
    batches = _stream_document_batches(
        document_store=document_store,
        batch_size=batch_size,
        include_embeddings=validate_embeddings,
        filters=filters
    )

    for batch in batches:
        total_docs += len(batch)
        for doc in batch:
            doc_id = doc.id
            
            # 1. content is None
            if doc.content is None:
                content_none_ids.append(doc_id)
                invalid_doc_ids.add(doc_id)
            else:
                # 2. Empty documents
                if doc.content == "":
                    empty_content_ids.append(doc_id)
                    invalid_doc_ids.add(doc_id)
                
                # 3. Very short chunks
                elif len(doc.content) < short_chunk_threshold:
                    short_chunk_ids.append(doc_id)
                    invalid_doc_ids.add(doc_id)
                
                # 4. Duplicate chunks (MD5 hash)
                content_hash = hashlib.md5(doc.content.encode("utf-8")).hexdigest()
                hash_map[content_hash].append(doc_id)

            # 5. Missing metadata fields
            if expected_metadata_keys:
                doc_meta = doc.meta or {}
                missing_keys = [
                    key for key in expected_metadata_keys 
                    if key not in doc_meta or doc_meta[key] is None
                ]
                if missing_keys:
                    missing_metadata_details.append({
                        "document_id": doc_id,
                        "missing_keys": missing_keys
                    })
                    invalid_doc_ids.add(doc_id)

            # 6. Null embeddings (only run if validate_embeddings is True)
            if validate_embeddings:
                if doc.embedding is None:
                    null_embedding_ids.append(doc_id)
                    invalid_doc_ids.add(doc_id)
                
                # 7. Embedding dimension mismatch
                else:
                    dim = len(doc.embedding)
                    dim_map[dim].append(doc_id)
                    if expected_embedding_dim is not None and dim != expected_embedding_dim:
                        dim_mismatch_details.append({
                            "document_id": doc_id,
                            "expected_dimension": expected_embedding_dim,
                            "actual_dimension": dim
                        })
                        invalid_doc_ids.add(doc_id)

    # Compile duplicates
    duplicates_list = []
    for content_hash, doc_ids in hash_map.items():
        if len(doc_ids) > 1:
            duplicates_list.append({
                "content_hash": content_hash,
                "document_ids": doc_ids
            })
            for d_id in doc_ids:
                invalid_doc_ids.add(d_id)

    # If there are multiple distinct embedding dimensions in the store, that's an internal mismatch
    if validate_embeddings and len(dim_map) > 1:
        # Flag all documents with non-majority dimension
        sorted_dims = sorted(dim_map.items(), key=lambda x: len(x[1]), reverse=True)
        majority_dim = sorted_dims[0][0]
        for dim, doc_ids in sorted_dims[1:]:
            for d_id in doc_ids:
                dim_mismatch_details.append({
                    "document_id": d_id,
                    "expected_dimension": majority_dim,
                    "actual_dimension": dim,
                    "reason": "Dimension differs from majority of store embeddings"
                })
                invalid_doc_ids.add(d_id)

    # Compute status strings
    status_content_none = "pass" if not content_none_ids else "fail"
    status_empty_content = "pass" if not empty_content_ids else "fail"
    status_short_chunks = "pass" if not short_chunk_ids else "warning"
    status_duplicates = "pass" if not duplicates_list else "fail"
    status_missing_metadata = "pass" if not missing_metadata_details else "warning"
    
    if not validate_embeddings:
        status_null_embeddings = "skipped"
        status_dim_mismatch = "skipped"
    else:
        status_null_embeddings = "pass" if not null_embedding_ids else "fail"
        status_dim_mismatch = "pass" if not dim_mismatch_details else "fail"

    total_issues = (
        len(content_none_ids)
        + len(empty_content_ids)
        + len(short_chunk_ids)
        + sum(len(dup["document_ids"]) for dup in duplicates_list)
        + len(missing_metadata_details)
        + len(null_embedding_ids)
        + len(dim_mismatch_details)
    )

    report = {
        "summary": {
            "total_documents": total_docs,
            "valid_documents": total_docs - len(invalid_doc_ids),
            "invalid_documents": len(invalid_doc_ids),
            "total_issues_found": total_issues,
        },
        "checks": {
            "content_none": {
                "status": status_content_none,
                "count": len(content_none_ids),
                "document_ids": content_none_ids,
            },
            "empty_content": {
                "status": status_empty_content,
                "count": len(empty_content_ids),
                "document_ids": empty_content_ids,
            },
            "short_chunks": {
                "status": status_short_chunks,
                "count": len(short_chunk_ids),
                "document_ids": short_chunk_ids,
                "threshold": short_chunk_threshold,
            },
            "duplicate_chunks": {
                "status": status_duplicates,
                "count": len(duplicates_list),
                "duplicates": duplicates_list,
            },
            "missing_metadata": {
                "status": status_missing_metadata,
                "count": len(missing_metadata_details),
                "details": missing_metadata_details,
            },
            "null_embeddings": {
                "status": status_null_embeddings,
                "count": len(null_embedding_ids),
                "document_ids": null_embedding_ids,
            },
            "embedding_dimension_mismatch": {
                "status": status_dim_mismatch,
                "count": len(dim_mismatch_details),
                "expected_dimension": expected_embedding_dim,
                "actual_dimensions": {str(k): len(v) for k, v in dim_map.items()} if validate_embeddings else {},
                "details": dim_mismatch_details,
            },
        },
    }

    return report
