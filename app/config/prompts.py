RAG_SYSTEM_PROMPT = """
You are an enterprise knowledge assistant.

Use only the provided context.
If the answer is not in the context, say so.
"""

QUERY_REWRITE_PROMPT = """
Rewrite the query for better retrieval.

Query:
{query}
"""