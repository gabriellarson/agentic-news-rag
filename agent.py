import lmstudio as lms
from qdrant_client import QdrantClient, models
from typing import List
import json

input = "whats going on with unilever?"

client = QdrantClient(url="http://localhost:6333")

model = lms.llm("qwen3-30b-a3b")
embedding_model = lms.embedding_model("text-embedding-qwen3-embedding-0.6b")

class query_schema(lms.BaseModel):
    queries: List[str]

class relevance_schema(lms.BaseModel):
    relevant: bool

class event(lms.BaseModel):
    entity: str
    description: str
    time: str

class event_schema(lms.BaseModel):
    events: List[event]

generate_query_prompt = "Generate useful queries for semantically searching a database of news articles that will lead to information that answers the user's question. Respond only with json containing the list of queries. User's input question: " + input
response = model.respond(generate_query_prompt, response_format=query_schema)
queries = json.loads(response.content)["queries"]

articles = []
seen_ids = set()
for q in queries:
    query_vector = embedding_model.embed(q)
    points = client.query_points(
        collection_name="article-collection",
        query = query_vector,
        score_threshold = 0.1,
    ).points

    for point in points:
        if point.id not in seen_ids:
            seen_ids.add(point.id)
            articles.append(point)


#with all articles retrieved, ask if relevant to the intial user question
relevant_articles = []
for article in articles:
    question_relevance_prompt = F"""Is this article relevant (even tangentially) to the user's input? Answer true or false.
User's input: {input}
Article Title: {article.payload["title"]}
Subtitle: {article.payload["subtitle"]}
Content: {article.payload["content"]}
"""
    response = model.respond(question_relevance_prompt, response_format=relevance_schema)
    #print(article.payload["title"], response.content)
    if json.loads(response.content)["relevant"]: relevant_articles.append(article)


#all relevant articles get events extracted from them
events = []
for article in relevant_articles:
    extract_events_prompt = f"""Extract all notable events from this news article.
For each event, provide:
- entity: The main person, company, organization, or thing involved
- description: A concise description of what happened (one sentence)
- time: When it happened (can be exact dates like "2024-03-15", relative times like "last week", or periods like "Q4 2023")

Focus on events that could be related to the user's input: {input}

Article Title: {article.payload["title"]}
Subtitle: {article.payload["subtitle"]}
Content: {article.payload["content"]}
"""
    response = model.respond(extract_events_prompt, response_format=event_schema)
    #print(response)



#resolve extracted events to exact dates

#construct timeline

#generate report from timeline


    


