import os
from googleapiclient.discovery import build
import googleapiclient.errors

class WebSearchAgent:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("CUSTOM_SEARCH_ENGINE_ID")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        if not self.cse_id:
            raise ValueError("CUSTOM_SEARCH_ENGINE_ID environment variable not set.")
        self.search_service = build("customsearch", "v1", developerKey=self.api_key)

    def run(self, task):
        yield f"[WebSearchAgent] Searching the web for: {task}"
        try:
            search_results = self.search_service.cse().list(
                q=task,
                cx=self.cse_id,
                num=3
            ).execute()
            if 'items' in search_results:
                for item in search_results['items']:
                    yield f"[WebSearchAgent] {item.get('title')}: {item.get('link')}"
            else:
                yield "[WebSearchAgent] No results found."
        except Exception as e:
            yield f"[WebSearchAgent] Error: {e}"
