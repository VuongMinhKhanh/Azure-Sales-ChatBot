import requests

# Chatwoot API Configuration
BASE_URL = "https://app.chatwoot.com"  # Replace with your instance URL
API_TOKEN = "Bra2r76fpTm2NjKQgsBrw1r9"
ACCOUNT_ID = "110405"


def send_message(conversation_id, content):
    url = f"{BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/conversations/{conversation_id}/messages"
    headers = {
        "api_access_token": f"{API_TOKEN}",
    }
    payload = {
        "content": content,
        "message_type": "incoming"
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        print("Message sent successfully!")
    else:
        print(f"Failed to send message: {response.status_code}, {response.text}")


# Example Usage
conversation_id = "1"  # Replace with the conversation ID
bot_response = "Im ken"
send_message(conversation_id, bot_response)
