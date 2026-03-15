

import asyncio
from src.core.client.embedding_rerank_client import LocalEmbeddingModel

async def test_local_embedding():
    print("=== 測試 LocalEmbeddingModel ===\n")
    
    # 初始化模型
    print("正在初始化 LocalEmbeddingModel...")
    embedding_model = LocalEmbeddingModel(
        model_path='/data/data_science_department/sharefolder/eddie/llm_ft/bert_base_uncased'  # 指定本地模型路徑
    )
    print("模型初始化完成\n")
    
    # 測試 embed_query
    print("=== 測試 embed_query ===")
    test_query = "這是一個測試查詢"
    print(f"查詢: {test_query}")
    query_embedding = await embedding_model.embed_query(test_query)
    print(f"嵌入向量維度: {len(query_embedding)}")
    print(f"嵌入向量前5個值: {query_embedding[:5]}\n")
    
    # 測試 embed_documents
    print("=== 測試 embed_documents ===")
    test_documents = [
        "這是第一個文檔",
        "這是第二個文檔",
        "這是第三個文檔"
    ]
    print(f"文檔數量: {len(test_documents)}")
    for i, doc in enumerate(test_documents):
        print(f"  文檔 {i+1}: {doc}")
    
    doc_embeddings = await embedding_model.embed_documents(test_documents)
    print(f"\n生成的嵌入向量數量: {len(doc_embeddings)}")
    for i, emb in enumerate(doc_embeddings):
        print(f"  文檔 {i+1} 嵌入向量維度: {len(emb)}")
        print(f"  文檔 {i+1} 嵌入向量前5個值: {emb[:5]}")
    
    print("\n=== 測試完成 ===")

if __name__ == "__main__":
    asyncio.run(test_local_embedding())
