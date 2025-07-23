from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

collections_list = client.get_collections()

if not any(c.name == "article-collection" for c in collections_list.collections):
    client.create_collection(collection_name="article-collection", vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE))
    print("Created collection")
else:
    print("Collection name already in use")

