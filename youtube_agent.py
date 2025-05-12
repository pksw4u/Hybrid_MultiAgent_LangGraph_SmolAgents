import os
import re

from googleapiclient.discovery import build
import googleapiclient.errors

class YouTubeAgent:
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("CUSTOM_SEARCH_ENGINE_ID")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        if not self.cse_id:
            raise ValueError("CUSTOM_SEARCH_ENGINE_ID environment variable not set.")
        self.search_service = build("customsearch", "v1", developerKey=self.api_key)
        self.youtube_service = build("youtube", "v3", developerKey=self.api_key)

    def run(self, task):
        yield f"[YouTubeAgent] Searching for YouTube videos about: {task}"
        try:
            search_results = self.search_service.cse().list(
                q=task + " site:youtube.com",
                cx=self.cse_id,
                num=3
            ).execute()
            video_results = []
            if 'items' in search_results:
                for item in search_results['items']:
                    if 'youtube.com/watch?v=' in item.get('link', ''):
                        video_results.append({
                            'title': item.get('title'),
                            'url': item.get('link')
                        })
            if not video_results:
                yield "[YouTubeAgent] No relevant YouTube videos found."
                return
            for video in video_results:
                yield f"[YouTubeAgent] Title: {video['title']}\nURL: {video['url']}"
                video_id = self.extract_video_id(video['url'])
                if video_id:
                    details = self.get_video_details([video_id])
                    if details and video_id in details:
                        d = details[video_id]
                        yield f"[YouTubeAgent] Video Details: Title: {d.get('title')}, Duration: {d.get('duration')}, Published: {d.get('publishedAt')}"
        except Exception as e:
            yield f"[YouTubeAgent] Error: {e}"

    @staticmethod
    def extract_video_id(youtube_url):
        if youtube_url:
            match = re.search(r"v=([^&]+)", youtube_url)
            if match:
                return match.group(1)
        return None

    def get_video_details(self, video_ids):
        if not video_ids:
            return {}
        try:
            ids_string = ",".join(video_ids)
            response = self.youtube_service.videos().list(
                part="snippet,contentDetails",
                id=ids_string
            ).execute()
            video_details = {}
            if 'items' in response:
                for item in response['items']:
                    video_details[item['id']] = {
                        'title': item['snippet']['title'],
                        'description': item['snippet']['description'],
                        'publishedAt': item['snippet']['publishedAt'],
                        'duration': item['contentDetails']['duration']
                    }
            return video_details
        except Exception as e:
            return {}
