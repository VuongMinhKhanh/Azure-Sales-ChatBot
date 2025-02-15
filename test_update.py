import json

def get_row_by_id(weaviate_client, vector_id, collection_name):
    # Query Weaviate to find the row corresponding to the given ID
    query = f"""
    {{
      Get {{
        {collection_name}(
          where: {{
            path: ["text"],
            operator: Equal,
            valueString: "{vector_id}"
          }}
        ) {{
          row
        }}
      }}
    }}
    """
    response = weaviate_client.graphql_raw_query(query)
    results = response.get[collection_name]

    if results:
        return results[0]['row']
    else:
        print(f"No row found for ID: {vector_id}")
        return None


def get_all_columns_by_row(weaviate_client, row, collection_name):
    # Query Weaviate to retrieve all columns for the specified row
    query = f"""
    {{
      Get {{
        {collection_name}(
          where: {{
            path: ["row"],
            operator: Equal,
            valueInt: {row}
          }}
          limit: 10000
        ) {{
          column
          text
          _additional {{
            id
          }}
        }}
      }}
    }}
    """
    response = weaviate_client.graphql_raw_query(query)
    results = response.get[collection_name]
    return results


def adjust_columns_by_patch_data(weaviate_client, patch_data, collection_name):
    # Step 1: Get the row from the patch_data
    vector_id = patch_data["id"]  # Ensure `id` exists in patch_data
    if not vector_id:
        print("Error: 'id' not found in patch_data")
        return

    row = get_row_by_id(weaviate_client, vector_id, collection_name)  # Use the earlier function
    if row is None:
        print(f"Row not found for ID: {vector_id}")
        return

    # Step 2: Retrieve all data for the row
    results = get_all_columns_by_row(weaviate_client, row, collection_name)
    print(results)

    # Step 3: Adjust only columns specified in patch_data
    adjusted_data = []
    for res in results:
        column = res['column']
        text = res['text']

        # Skip adjustment for columns like 'id', 'Nội dung', or 'Mô tả'
        if column in ['id', 'Nội dung', 'Mô tả']:
            continue

        # Check if this column is in the patch_data keys
        if column in patch_data:
            # Adjust only the value after the column name
            column_prefix = f"{column}: "
            if text.startswith(column_prefix):  # Ensure the format matches
                new_value = patch_data[column]
                adjusted_text = f"{column_prefix}{new_value}"
                adjusted_data.append({
                    "column": column,
                    "adjusted_text": adjusted_text,
                    "id": res['_additional']['id']
                })

    # Step 4: Update adjusted columns back into the vector database
    for data in adjusted_data:
        update_vector_column(weaviate_client, data['id'], data['adjusted_text'], collection_name)

    print("Adjustments completed successfully.")
    print(adjusted_data)
    return adjusted_data


def update_vector_column(weaviate_client, vector_id, new_text, collection_name):
    """
        Updates a vector's properties in Weaviate using its UUID.

        :param weaviate_client: The Weaviate client instance.
        :param uuid: The UUID of the vector to update.
        :param new_properties: A dictionary containing the properties to update.
        """
    try:
        # Update the vector in Weaviate
        chatbot_collection = weaviate_client.collections.get(collection_name)
        chatbot_collection.data.update(
            uuid=vector_id,
            properties={
                "text": new_text
            }
        )
        print(f"Update response: {new_text}")
        return new_text
    except Exception as e:
        print(f"Error updating vector with UUID {vector_id}: {e}")
        raise e

# # Example patch data
# patch_data = {
#     "nhanhieu_en": "Chinh hãng1",
#     "ngaysua": "1737608154",
#     "id": "2",
#     "Bán chạy": 0,
#     "Tên": "Loa đẹp vkl"
# }

# import os
# import weaviate
# from weaviate.classes.init import Auth
# from dotenv import load_dotenv
#
# # Load environment variables from .env file
# load_dotenv()
#
# weaviate_url = os.getenv("WEAVIATE_URL")
# weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
# openai_api_key = os.getenv("OPENAI_API_KEY")
#
# weaviate_client = weaviate.connect_to_weaviate_cloud(
#                 cluster_url=weaviate_url,
#                 auth_credentials=Auth.api_key(weaviate_api_key),
#                 headers={"X-OpenAI-Api-Key": openai_api_key}
#             )
#
# # Call the function
# collection_name = "ChatBot"
# adjust_columns_by_patch_data(weaviate_client, patch_data, collection_name)