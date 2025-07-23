from qdrant_client import QdrantClient, models
import json

config = json.load(open("config.json", 'r'))

client = QdrantClient(url=config["qdrant_client_url"])

collections_list = client.get_collections()

if not any(c.name == config["qdrant_collection_name"] for c in collections_list.collections):
    client.create_collection(collection_name=config["qdrant_collection_name"], vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE))
    print("Created collection")
else:
    print("Collection name already in use")

