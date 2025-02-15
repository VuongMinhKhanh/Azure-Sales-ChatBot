from collections import defaultdict
import requests
from flask import Flask, render_template, jsonify, request
from langchain_core.messages import HumanMessage, AIMessage
from Sales_Consulting_Chatbot import *
from session_control import *
from dotenv import load_dotenv
import os
from test_update import adjust_columns_by_patch_data
import json
import redis

# Load environment variables from .env file
load_dotenv()

# Flask App Initialization
application = Flask(__name__)

# Chatwoot API Configuration
BASE_URL = "https://app.chatwoot.com"  # Replace with your instance URL
API_TOKEN = os.getenv("CHATWOOT_API_TOKEN")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
INBOX_ID = os.getenv("INBOX_ID")
AGENT_ID = os.getenv("AGENT_ID")
REDIS_PASS = os.getenv("REDIS_PASS")

# Global variables
processed_messages = defaultdict(lambda: None)
MESSAGE_PROCESSING_TIMEOUT = 5
collection_name = os.getenv("COLLECTION_NAME")

# Redis Client
try:
    redis_client = redis.Redis(host='redis-19781.c295.ap-southeast-1-1.ec2.redns.redis-cloud.com', password=REDIS_PASS, port=19781, db=0)
    redis_client.ping()
    print("Connected to Redis successfully!")
except redis.ConnectionError as e:
    print(f"Redis connection failed: {e}")
@application.route('/')
def hello_world():
    return render_template('index.html')

@application.route('/test-api', methods=['POST', 'PATCH'])
def test_api():
    if request.content_type == 'application/json':
        data = request.get_json()
    elif request.content_type == 'application/x-www-form-urlencoded':
        data = request.form.to_dict()
    else:
        return jsonify({"status": "error", "message": "Unsupported Media Type"}), 415

    print(data)
    print(data.get("id"))
    adjust_columns_by_patch_data(weaviate_client, data, collection_name)

    return jsonify({
        "status": "success",
        "message": "Data received successfully!",
        "received_data": data
    }), 200


@application.route("/api/webhook", methods=["POST"])
def webhook():
    try:
        body = request.json
        # print("Received payload:", json.dumps(body, indent=2))
        event_type = body.get("event")
        print(f"Received event: {event_type}")

        conversation = body.get("conversation", {})
        messages = conversation.get("messages", [])

        # conversation_id = conversation.get("id")
        # print(f"Conversation ID: {conversation_id}")

        if messages and event_type == "message_created":
            print("Messages received")
            last_message = messages[-1]
            print(f"Last message: {last_message}")

            conversation_id = last_message.get("conversation_id")
            print(f"Conversation ID: {conversation_id}")

            content = last_message.get("content")
            print(f"Content: {content}")

            sender_type = last_message.get("sender_type")
            print(f"Sender Type: {sender_type}")

            assignee_id = last_message.get("conversation", {}).get("assignee_id", None)
            print(f"Assignee ID: {assignee_id}")

        else:
            print("No messages found in this event.")
            conversation_id, content, sender_type, assignee_id = None, None, None, None

        if event_type == "conversation_status_changed":
            if body.get("status") == "resolved":
                conversation_id = body.get("id")
                if id:
                    send_message_to_chatwoot(conversation_id, os.getenv("RESET_TEXT"))
                else:
                    send_message_to_chatwoot(conversation_id, os.getenv("BACK_TO_BOT"))

                print(f"Conversation {conversation_id} marked as resolved.")
                remove_assigned(conversation_id)
                clear_chat_history(conversation_id)
                set_unassigned(conversation_id)  # Assign the conversation back to None

                return jsonify({"status": "success", "message": "Conversation resolved and unassigned."}), 200
            else:
                print("Conversation ID not found.")
                return jsonify({"status": "error", "message": "Conversation ID not found."}), 200

        # Ignore the message if it's not from a user
        if sender_type != "Contact":
            return jsonify({"status": "ignored", "message": "Message is not from a user."}), 200

        # Ignore the message if the conversation is assigned to a consultant
        if assignee_id is not None:
            print(f"Conversation {conversation_id} is assigned to consultant {assignee_id}. Chatbot is disabled.")
            return jsonify({"status": "ignored", "message": "Chatbot is disabled for assigned conversation."}), 200

        # Ignore the message if content or conversation_id is missing
        if not content or not conversation_id:
            return jsonify({"status": "ignored", "message": "Missing content or conversation_id."}), 200

        # Process the message using the chatbot
        chat_history = get_chat_history(conversation_id)
        chat_history.append(HumanMessage(content=content))

        if content.lower() in ["talk to consultant", "need human help", "consultant please", "tư vấn viên"]:
            send_message_to_chatwoot(conversation_id, os.getenv("ASSIGNING_TEXT"))
            assign_to_consultant(conversation_id, AGENT_ID)
            mark_assigned(conversation_id, AGENT_ID)
            return jsonify({"status": "success", "message": "Assigned to consultant"}), 200

        # Process the message using the chatbot
        qa = initialize_rag(llm, data, retriever, chat_history)
        result = qa.invoke({"input": content, "chat_history": chat_history})
        chat_history.append(AIMessage(content=result["answer"]))
        store_chat_history(conversation_id, chat_history)
        send_message_to_chatwoot(conversation_id, result["answer"])

        return jsonify({"status": "success", "message": "Chatbot response processed."}), 200

    except Exception as e:
        print(f"Unexpected error in webhook: {e}")
        return jsonify({"status": "error", "details": str(e)}), 500


def send_message_to_chatwoot(conversation_id, message_content):
    url = f"{BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/conversations/{conversation_id}/messages"
    headers = {"api_access_token": API_TOKEN}
    data = {"content": message_content}
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        print("Message sent successfully to Chatwoot.")
    else:
        print(f"Failed to send message: {response.status_code}, {response.text}")


def set_unassigned(conversation_id):
    url = f"{BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/conversations/{conversation_id}/assignments"
    headers = {"api_access_token": API_TOKEN}
    data = {"assignee_id": None}
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        print(f"Conversation {conversation_id} successfully set to Unassigned status.")
    else:
        print(f"Failed to set conversation to Unassigned status: {response.status_code}, {response.text}")


def assign_to_consultant(conversation_id, consultant_id):
    """
    Assign a conversation to a specific consultant if it is not already assigned.

    :param conversation_id: The ID of the conversation to assign.
    :param consultant_id: The ID of the consultant to assign the conversation to.
    :return: Boolean indicating success or failure.
    """
    url = f"{BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/conversations/{conversation_id}/assignments"
    headers = {"api_access_token": API_TOKEN}
    data = {"assignee_id": consultant_id}

    # Check if the conversation is already assigned to avoid conflicts
    check_url = f"{BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/conversations/{conversation_id}"
    response_check = requests.get(check_url, headers=headers)

    if response_check.status_code == 200:
        current_assignee = response_check.json().get("assignee_id")
        if current_assignee is not None:
            print(f"Conversation {conversation_id} is already assigned to consultant {current_assignee}. No reassignment performed.")
            return False  # Already assigned

    # Proceed to assign the consultant
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        print(f"Conversation {conversation_id} successfully assigned to consultant {consultant_id}.")
        return True
    else:
        print(f"Failed to assign consultant: {response.status_code}, {response.text}")
        return False


def clear_chat_history(conversation_id):
    """Clear chat history from Redis."""
    redis_client.delete(f"chat_history:{conversation_id}")
    print(f"Cleared chat history for conversation_id: {conversation_id}")


def serialize_chat_history(chat_history):
    serialized_history = []
    for message in chat_history:
        print("type", message, type(message), isinstance(message, HumanMessage))
        if isinstance(message, HumanMessage):
            serialized_history.append({
                "type": "HumanMessage",
                "content": message.content
            })
        else:  # For other message types
            serialized_history.append({
                "type": "AiMessage",
                "content": message.content
            })
    return serialized_history


def store_chat_history(conversation_id, chat_history):
    """Store chat history in Redis."""
    chat_history_serialized = serialize_chat_history(chat_history)
    # print("chat_history_serialized", chat_history_serialized)
    redis_client.set(f"chat_history:{conversation_id}", json.dumps(chat_history_serialized))


def get_chat_history(conversation_id):
    """Retrieve chat history from Redis."""
    history = redis_client.get(f"chat_history:{conversation_id}")
    if history:
        deserialized_history = json.loads(history)
        return [
            HumanMessage(content=msg["content"]) if msg["type"] == "human" else AIMessage(content=msg["content"])
            for msg in deserialized_history
        ]
    return []


def is_assigned(conversation_id):
    """Check if a conversation is assigned to a consultant."""
    return redis_client.exists(f"assigned_conversation:{conversation_id}")


def mark_assigned(conversation_id, consultant_id):
    """Mark a conversation as assigned in Redis with a TTL of 1 hour."""
    redis_client.setex(f"assigned_conversation:{conversation_id}", 3600, consultant_id)


def remove_assigned(conversation_id):
    """Remove assigned status for a conversation."""
    redis_client.delete(f"assigned_conversation:{conversation_id}")


if __name__ == '__main__':
    application.run(debug=True, port=8000)
