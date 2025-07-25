import lmstudio as lms
from qdrant_client import QdrantClient, models
from typing import List
import json
from datetime import datetime
from pathlib import Path

class QuerySchema(lms.BaseModel):
    queries: List[str]
def generate_queries(input, model):
    generate_query_prompt = "Generate useful queries for semantically searching a database of news articles that will lead to information that answers the user's question. Respond only with json containing the list of queries. User's input question: " + input
    response = model.respond(generate_query_prompt, response_format=QuerySchema)
    queries = json.loads(response.content)["queries"]
    return queries

def retrieve_articles(queries, client, embedding_model):
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
    return articles

class RelevanceSchema(lms.BaseModel):
    relevant: bool
def filter_articles(articles, input, model):
    relevant_articles = []
    for article in articles:
        question_relevance_prompt = F"""Is this article relevant (even tangentially) to the user's input? Answer true or false.
User's input: {input}
Article Title: {article.payload["title"]}
Subtitle: {article.payload["subtitle"]}
Content: {article.payload["content"]}
"""
        response = model.respond(question_relevance_prompt, response_format=RelevanceSchema)
        #print(article.payload["title"], response.content)
        if json.loads(response.content)["relevant"]: relevant_articles.append(article)
    return relevant_articles

class Event(lms.BaseModel):
    entity: str
    description: str
class EventSchema(lms.BaseModel):
    events: List[Event]
def extract_events(relevant_articles, model):
    #all relevant articles get events extracted from them
    articles_events = []
    for article in relevant_articles:
        extract_events_prompt = f"""Extract all notable events from this news article.
For each event, provide:
- entity: The main person, company, organization, or thing involved
- description: A concise description of what happened, and when it happened

Focus on events that could be related to the user's input: {input}

Article Title: {article.payload["title"]}
Subtitle: {article.payload["subtitle"]}
Content: {article.payload["content"]}
"""
        response = model.respond(extract_events_prompt, response_format=EventSchema)
        articles_events.append([article, json.loads(response.content)])
    return articles_events

class TimestampSchema(lms.BaseModel):
    start_timestamp: datetime
    end_timestamp: datetime
def resolve_timestamps(articles_events, model):
    timestamped_articles_events = []
    for article_events in articles_events:
        article = article_events[0]
        events = article_events[1]
        timestamped_events = []

        for event in events['events']:
            resolve_timestamp_prompt = f"""Resolve the exact ISO 8601 (i.e. "2024-01-11T12:42:05.955Z") timestamps for the given article event, and article publication date.
Be as precise as possible when resolving the date. If no date for an event can be deduced, assume it happened at the time of publication.
If the event happened instantaneously, the start_timestamp and end_timestamp will have the exact same value. If it happened over a range of time, they will have different values that cover the range of time.

For example:
Article Publication: 2024-01-11T12:42:05.955Z
Event: 'Unilever is increasing brand and marketing spend, from 13% of sales in 2022 to 14.3% in 2023, to support growth.'
start_timestamp = 2022-01-01T00:00:00.000Z
end_timestamp = 2023-12-31T23:59:59.999Z

Example 2:
Article Publication: 2025-05-23T02:21:48.955Z
Event: 'Regulators warn of greenwashing in ESG ETFs, urging more due diligence on methodologies.'
start_timestamp = 2025-05-23T02:21:48.955Z
end_timestamp = 2025-05-23T02:21:48.955Z

Actual event to resolve:
Article Publication: {article.payload["published"]}
Event: {event["description"]}
"""
            response = model.respond(resolve_timestamp_prompt, response_format=TimestampSchema)
            timestamp_data = json.loads(response.content)
            
            timestamped_event = {
                "entity": event["entity"],
                "description": event["description"],
                "start_timestamp": timestamp_data["start_timestamp"],
                "end_timestamp": timestamp_data["end_timestamp"],
                "article_title": article.payload["title"],
                "article_published": article.payload["published"]
            }
            timestamped_events.append(timestamped_event)
        
        timestamped_articles_events.append({
            "article": article,
            "events": timestamped_events
        })
    
    return timestamped_articles_events

def construct_timeline(timestamped_articles_events):
    timeline = []
    
    for article in timestamped_articles_events:
        for event in article["events"]:
            timeline.append(event)
    
    timeline.sort(key=lambda x: x["start_timestamp"])
    
    return timeline

def generate_report(timeline, input, model):
    if not timeline:
        return "No relevant events found for the given query."
    
    events_summary = []
    for event in timeline:
        event_text = f"""
Time: {event['start_timestamp']} to {event['end_timestamp']}
Entity: {event['entity']}
Event: {event['description']}
Source: {event['article_title']}"""
        events_summary.append(event_text)
    

    generate_report_prompt = f"""Based on the following chronological timeline of events, generate a comprehensive report that answers the user's question.

User's Question: {input}

Timeline of Events:
{chr(10).join(events_summary)}

Create a well-structured report that:
1. Directly addresses the user's question
2. Synthesizes the information from multiple events
3. Identifies key trends or patterns
4. Highlights the most significant findings
5. Provides a clear conclusion
6. Cites your relevant source articles

Format the report with clear sections and make it easy to read."""

    response = model.respond(generate_report_prompt)
    report = response.content
    
    report_with_metadata = f"""Timeline Analysis Report

Query: {input}
Generated: {datetime.now().isoformat()}
Total Events Analyzed: {len(timeline)}

---

{report}

---

## Event Timeline Details

"""
    
    for i, event in enumerate(timeline, 1):
        report_with_metadata += f"""
### Event {i}
- **Date:** {event['start_timestamp']} to {event['end_timestamp']}
- **Entity:** {event['entity']}
- **Description:** {event['description']}
- **Source:** {event['article_title']}
"""
    
    return report_with_metadata

    
if __name__ == "__main__":
    config = json.load(open("config.json", 'r'))
    input = config["input"]
    client = QdrantClient(url=config["qdrant_client_url"])
    model = lms.llm(config["lmstudio_llm"])
    embedding_model = lms.embedding_model(config["lmstudio_embedding"])

    print("GENERATING QUERIES")
    queries = generate_queries(input, model)

    print("RETRIEVING ARTICLES")
    articles = retrieve_articles(queries, client, embedding_model)

    print("FILTERING IRRELEVANT ARTICLES")
    relevant_articles = filter_articles(articles, input, model)

    print("EXTRACTING EVENTS")
    articles_events = extract_events(relevant_articles, model)

    print("RESOLVING TIMESTAMPS")
    timestamped_articles_events = resolve_timestamps(articles_events, model)

    print("CONSTRUCTING TIMELINE")
    timeline = construct_timeline(timestamped_articles_events)

    print ("GENERATING REPORT")
    report = generate_report(timeline, input, model)
    print(report)



