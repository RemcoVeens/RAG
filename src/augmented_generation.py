from src.hybrid_search import HybridSearch


class RAG(HybridSearch):
    def __init__(self):
        super().__init__()

    def generate(self, query: str):
        results = self.rrf_search(query, 60, 5)
        prompt = f"""Answer the question or provide information based on the provided documents.
        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Query: {query}

        Documents:
        {", ".join([res.data.title for _, res in results])}

        Provide a comprehensive answer that addresses the query:"""
        response = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        print(response.text)

    def summarize(self, query: str, limit: int):
        results = self.rrf_search(query, 60, limit)
        prompt = f"""
        Provide information useful to this query by synthesizing information from multiple search results in detail.
        The goal is to provide comprehensive information so that users know what their options are.
        Your response should be information-dense and concise, with several key pieces of information about the genre,
        plot, etc. of each movie.
        This should be tailored to Hoopla users. Hoopla is a movie streaming service.
        Query: {query}
        Search Results:
        {", ".join([res.data.title for _, res in results])}
        Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
        """
        response = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

        print("Search Results:")
        for _, result in results:
            print(f"- {result.data.title}")
        print(f"LLM Summary:\n{response.text}")

    def citations(self, query: str, limit: int):
        results = self.rrf_search(query, 60, limit)
        prompt = f"""Answer the question or provide information based on the provided documents.

        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        If not enough information is available to give a good answer,
        say so but give as good of an answer as you can while citing the sources you have.

        Query: {query}

        Documents:
        {", ".join([res.data.title + ": " + res.data.document[:100] for _, res in results])}

        Instructions:
        - Provide a comprehensive answer that addresses the query
        - Cite sources using [1], [2], etc. format when referencing information
        - If sources disagree, mention the different viewpoints
        - If the answer isn't in the documents, say "I don't have enough information"
        - Be direct and informative

        Answer:"""

        response = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        print("Search Results:")
        for _, result in results:
            print(f"- {result.data.title}")
        print(f"LLM Answer:\n{response.text}")

    def questions(self, query: str, limit: int):
        results = self.rrf_search(query, 60, limit)
        prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Question: {query}

        Documents:
        {"\n".join([res.data.title + ": " + res.data.document for _, res in results])}

        Instructions:
        - Answer questions directly and concisely
        - Be casual and conversational
        - Don't be cringe or hype-y
        - Talk like a normal person would in a chat conversation

        Answer:"""

        response = self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        print("Search Results:")
        for _, result in results:
            print(f"- {result.data.title}")
        print(f"Answer:\n{response.text}")
