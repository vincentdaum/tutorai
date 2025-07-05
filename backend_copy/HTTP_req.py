"""
TutorAI API Client - Send HTTP requests to the FastAPI chat service
"""

import requests
import json
import time
from typing import Optional

class TutorAIClient:
    def __init__(self, base_url: str = "http://localhost:8006", api_token: str = "devtoken"):
        """
        Initialize the TutorAI client
        
        Args:
            base_url: Base URL of the FastAPI service
            api_token: Bearer token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    def chat(self, query: str, client_id: str = "test_client", conversation_id: str = "test_conv") -> Optional[str]:
        """
        Send a regular (non-streaming) chat request
        
        Args:
            query: The question/message to send
            client_id: Client identifier
            conversation_id: Conversation identifier
            
        Returns:
            The response text or None if error
        """
        url = f"{self.base_url}/api/v1/chat"
        
        payload = {
            "client_id": client_id,
            "conversation_id": conversation_id,
            "query": query
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result.get("answer", "")
            
        except requests.exceptions.RequestException as e:
            print(f"Error sending chat request: {e}")
            return None
    
    # def chat_stream(self, query: str, client_id: str = "test_client", conversation_id: str = "test_conv"):
    #     """
    #     Send a streaming chat request and yield tokens as they arrive
        
    #     Args:
    #         query: The question/message to send
    #         client_id: Client identifier
    #         conversation_id: Conversation identifier
            
    #     Yields:
    #         Individual tokens from the streaming response
    #     """
    #     url = f"{self.base_url}/api/v1/chat/stream"
        
    #     payload = {
    #         "client_id": client_id,
    #         "conversation_id": conversation_id,
    #         "query": query
    #     }
        
    #     try:
    #         response = requests.post(url, headers=self.headers, json=payload, stream=True, timeout=30)
    #         response.raise_for_status()
            
    #         for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
    #             if chunk:
    #                 yield chunk
                    
    #     except requests.exceptions.RequestException as e:
    #         print(f"Error sending streaming chat request: {e}")
    #         yield f"<error>{e}</error>"


def main():
    """
    Example usage of the TutorAI client
    """
    # Initialize client
    client = TutorAIClient(
        base_url="http://localhost:8006",
        api_token="devtoken"  # Make sure this matches your TUTORAI_API_TOKEN
    )
    
    # Test queries
    test_queries = [
        "Was ist Informatik?",
        "Kannst du mir bei meinem TUB Modul helfen?",
        "Wie funktioniert maschinelles Lernen?"
    ]
    
    print("=== Testing Regular Chat Endpoint ===")
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        
        start_time = time.time()
        response = client.chat(
            query=query,
            client_id="demo_client",
            conversation_id="demo_conversation"
        )
        end_time = time.time()
        
        if response:
            print(f"Response: {response}")
            print(f"Time taken: {end_time - start_time:.2f} seconds")
        else:
            print("Failed to get response")
    
    print("\n" + "="*50)

def debug_server_status():
    """
    Check if server is running and get detailed error info
    """
    print("\n=== Debug Server Status ===")
    
    # First, check if server is reachable
    try:
        response = requests.get("http://localhost:8006")
        print(f"Server is reachable - Status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server at localhost:8006")
        print("   Make sure the FastAPI service is running:")
        print("   uvicorn fastapi_chat_service_cpu:app --host 0.0.0.0 --port 8006 --reload")
        return
    except Exception as e:
        print(f"Connection issue: {e}")
        return
    
    # Try to get detailed error from the API
    url = "http://localhost:8006/api/v1/chat"
    headers = {
        "Authorization": "Bearer devtoken",
        "Content-Type": "application/json"
    }
    
    data = {
        "client_id": "debug_client",
        "conversation_id": "debug_conv",
        "query": "test"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        # Try to get error details
        try:
            error_details = response.json()
            print(f"Error Details: {json.dumps(error_details, indent=2)}")
        except:
            print(f"Raw Response Text: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


def simple_request_example():
    """
    Simple example using just the requests library
    """
    print("\n=== Simple Request Example ===")
    
    url = "http://localhost:8006/api/v1/chat"
    headers = {
        "Authorization": "Bearer devtoken",
        "Content-Type": "application/json"
    }
    
    data = {
        "client_id": "simple_client",
        "conversation_id": "simple_conv",
        "query": "Hallo, wie geht es dir?"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        print(f"Status Code: {response.status_code}")
        print(f"Response: {result}")
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Error response text: {e.response.text}")


if __name__ == "__main__":
    print("TutorAI API Client Test")
    print("Make sure your FastAPI service is running on http://localhost:8006")
    print("And that TUTORAI_API_TOKEN is set to 'devtoken'")
    
    # First debug the server
    debug_server_status()
    
    # Run the main demo
    main()
    
    # Run simple example
    simple_request_example()