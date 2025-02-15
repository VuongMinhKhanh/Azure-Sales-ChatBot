from threading import Timer
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Global variables
# weaviate_client = None
# global weaviate_client, is_connected, idle_timer, idle_timeout
weaviate_client = None
is_connected = False

import os
import weaviate
from weaviate.classes.init import Auth

weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Function to initialize Weaviate client
def connect_weaviate():
    global weaviate_client, is_connected
    try:
        if weaviate_client is None or not is_connected:
            weaviate_client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,
                auth_credentials=Auth.api_key(weaviate_api_key),
                headers={"X-OpenAI-Api-Key": openai_api_key}
            )
            is_connected = True
            print("Weaviate client connected.")
        # elif weaviate_client.is_closed():
        #     weaviate_client.connect()
        #     is_connected = True
        #     print("Reconnected Weaviate client.")
    except Exception as e:
        print(f"Error connecting to Weaviate: {e}")
        is_connected = False


# Function to close Weaviate client
def close_weaviate():
    global weaviate_client, is_connected
    if is_connected and weaviate_client is not None:
        weaviate_client.close()
        weaviate_client = None
        is_connected = False
        print("Closed Weaviate connection")

# Function to reset the idle timer
# def reset_idle_timer():
#     global idle_timer
#     if idle_timer is not None:
#         idle_timer.cancel()  # Cancel the previous timer
#     idle_timer = Timer(idle_timeout, handle_user_idle)
#     idle_timer.start()
#     print("Idle timer reset")

# Function to handle user idle timeout
def handle_user_idle():
    print("User is idle. Closing Weaviate connection...")
    close_weaviate()

