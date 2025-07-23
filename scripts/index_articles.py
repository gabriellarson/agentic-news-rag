from qdrant_client import QdrantClient, models
import uuid
import hashlib
from pathlib import Path
import lmstudio as lms
import json

config = json.load(open("config.json", 'r'))

client = QdrantClient(url=config["qdrant_client_url"])

embedding_model = lms.embedding_model(config["lmstudio_embedding"])

folder = Path(config["articles_path"])
txts = list(folder.glob("*.txt"))

def parse(txt):
    lines = open(txt, 'r', encoding='utf-8').readlines()
    payload = {
        "title": lines[0][6:].strip(),
        "subtitle": lines[1][9:].strip(),
        "authors": lines[2][8:].strip(),
        "published": lines[3][10:].strip(),
        "content": ''.join(lines[5:])
    }
    return payload

for txt in txts:

    txt_id = str(uuid.UUID(hashlib.md5(txt.name.encode('utf-8')).hexdigest())) # use article name for deterministic UUID
    if(len(client.retrieve(collection_name=config["qdrant_collection_name"], ids=[txt_id], with_payload=False, with_vectors=False)) > 0): # check if UUID exists already
        continue

    txt_payload = parse(txt)

    txt_vector = embedding_model.embed(f'{txt_payload["title"]}\n\n{txt_payload["subtitle"]}\n\n{txt_payload["content"]}')

    client.upsert(
        collection_name=config["qdrant_collection_name"],
        points=[
            models.PointStruct(
                id = txt_id,
                payload = txt_payload,
                vector = txt_vector
            )
        ]
    )