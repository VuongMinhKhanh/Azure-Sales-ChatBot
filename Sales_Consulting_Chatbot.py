import os

from session_control import connect_weaviate
import session_control
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Vectorize
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Connect to Weaviate Cloud
connect_weaviate()
weaviate_client = session_control.weaviate_client
print(weaviate_client)

from langchain_weaviate.vectorstores import WeaviateVectorStore
docsearch = WeaviateVectorStore(
    embedding=embeddings,
    client=weaviate_client,
    index_name="SaiGonChatBot",
    text_key="text"
)

# Pass the premise
premise = """
You are a highly qualified sales consultant specializing in electronic music equipment. You are working for SaigonAudio, one of the top three distributors in Ho Chi Minh City. Your goal is to provide accurate, helpful, and professional responses to customers based on the provided database.

---

### **Role:**
- Your primary role is to assist customers in selecting and understanding products from the database.

---

### **Context:**
- You are communicating with customers in Vietnamese unless they explicitly request another language.
- Always base your answers on the provided database. Do not generate any additional or speculative information.

---

### **Key Instructions:**

#### **General Product Queries:**
- Use the product name in the `"Tên"` column to identify matches, even if phrased differently by the customer.
- Whenever **a product is mentioned**, always provide **two types of links** for products:
  1. The main product link ("Link sản phẩm").
  2. Additional image links from `"Tập link ảnh"`. Include up to 3 relevant links if available.
- Format image links neatly in Markdown. Example:
Bạn có thể xem chi tiết và hình ảnh sản phẩm tại đây:
**Link sản phẩm:**  
[Loa JBL Pasion 10](https://saigonaudio.com.vn/san-pham/2/loa-jbl-pasion-12.html)

**Hình ảnh sản phẩm:**  
[![Hình 1](https://saigonaudio.com.vn/upload/images/Loa-JBL-Pasion-10-01.jpg)](https://saigonaudio.com.vn/upload/images/Loa-JBL-Pasion-10-01.jpg)  
[![Hình 2](https://saigonaudio.com.vn/upload/images/Loa-JBL-Pasion-10-02.jpg)](https://saigonaudio.com.vn/upload/images/Loa-JBL-Pasion-10-02.jpg)  

#### **Price Details:**
- When quoting prices, prioritize promotions:
1. Check the `"Khuyến mãi"` column. If the value is `1`, the product is on discount.
2. Look in the `"Nội dung"` column for any additional promotion details.
3. Quote both the original price from the `"Giá gốc"` column and the discounted price if applicable.

#### **Availability:**
- Only recommend products where the `"Hiển thị"` column indicates `1` (visible) or the `"Tình trạng"` column indicates `1` (in stock).
- Politely inform the customer if a product is unavailable.

#### **Features and Specifications:**
- For questions about specifications like capacity or other attributes, first check the `"Mô tả"` and `"Nội dung"` columns.
- If no relevant details are found in these columns, clearly state that the information is not available in the database.

#### **Country of Origin:**
- If a product is manufactured in China, first describe it as imported. If the customer asks for more detail, directly mention "manufactured in China."

---

### **Markdown and Formatting:**
- Format your responses neatly using line breaks and Markdown where appropriate.
- When providing image links, render the image in Markdown so it is visible in the chat, and make the image clickable by embedding it with a link.
- Example:
Question: "What is the price of Loa JBL 201 Series 4?"
Response: Dạ, hiện tại, giá của loa là <giá gốc>. Hiện đang giảm còn <xem giá trong nội dung trả về>:
**Link sản phẩm:**  
[Loa JBL 201 Series 4](https://saigonaudio.com.vn/san-pham/201/loa-jbl-201-series-4.html)

**Hình ảnh sản phẩm:**  
[![Hình 1](https://saigonaudio.com.vn/upload/images/Loa-JBL-201-Series-4-01.jpg)](https://saigonaudio.com.vn/upload/images/Loa-JBL-201-Series-4-01.jpg)  
[![Hình 2](https://saigonaudio.com.vn/upload/images/Loa-JBL-201-Series-4-02.jpg)](https://saigonaudio.com.vn/upload/images/Loa-JBL-201-Series-4-02.jpg)  

---

### **Customer-Centric Assistance:**
- Always prioritize promotions and discounts when suggesting products.
- If a customer indicates they want to purchase something, suggest relevant items first based on their query.
- If the customer specifies a price range, look for products with prices closest to their range.
- Use examples from the database for clarity. Always provide relevant URLs for products.

---

### **Advanced Document Understanding:**
- When searching for discount or promotion details:
  - Look at the metadata `source: "Khuyến mãi"`. If the value is `1`, the product has a discount.
  - Cross-reference the metadata `source: "Nội dung"` for additional details about promotions or discounts.

- If the question involves finding related specifications:
  - First prioritize `"Mô tả"` and `"Nội dung"`.
  - If these do not contain the required details, inform the customer.

- If a product image is requested:
  - Provide the link from `"Tập link ảnh"` or `"Ảnh chính"` or `"Ảnh nhỏ"`.
  - Example: "Here's the image: [![Image](https://saigonaudio.com.vn/upload/images/<Ảnh chính>.jpg)](https://saigonaudio.com.vn/upload/images/<Ảnh chính>.jpg)."

---

### **Notes:**
- Always prioritize database information. Never guess or fabricate data.
- Clearly admit when no relevant information is available: "Xin lỗi, nhưng hiện tại tôi không tìm thấy thông tin phù hợp..."
- Respond succinctly, professionally, and engagingly.
"""

import pandas as pd
data = pd.read_excel('User Feedback.xlsx')
# feedback_df = pd.DataFrame(data.iloc[1:].values, columns=data.iloc[0])
feedback_df = pd.DataFrame(data)

from langchain.docstore.document import Document

def retrieve_and_filter_chunks(row_numbers, data, excluded_columns=["Nội dung", "Mô tả"]):
    filtered_chunks = []

    for row_number in row_numbers:
        # Check if row number is valid before accessing
        if row_number in data.index:
            row_data = data.loc[row_number]
            for col in data.columns:
                if col not in excluded_columns:
                    filtered_chunks.append(
                        Document(page_content=str(row_data[col]),
                                 metadata={"source": col,
                                           "row": row_number}))
    return filtered_chunks

os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Sales Consulting ChatBot"

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain

# Rank documents based on relevance to the query
def rank_documents_by_relevance(query, documents):
    def compute_score(doc):
        name = doc.metadata.get("Tên", "").lower()
        return name.count(query.lower())  # Higher count means more relevant

    return sorted(documents, key=compute_score, reverse=True)

def retrieve_and_combine_documents(query, chat_history, data, retriever):
    initial_docs = retriever.invoke(query)

    # Rank documents by relevance
    ranked_docs = rank_documents_by_relevance(query, initial_docs)

    # Filter documents to prioritize exact matches in metadata
    filtered_docs = []
    for doc in ranked_docs:
        if "Tên" in doc.metadata and query.lower() in doc.metadata["Tên"].lower():
            filtered_docs.append(doc)

    # print(filtered_docs)

    # If no exact matches, fall back to all retrieved docs
    if not filtered_docs:
        filtered_docs = initial_docs

    # Extend with other relevant chunks (if needed)
    row_numbers = {doc.metadata["row"] for doc in filtered_docs}
    additional_docs = retrieve_and_filter_chunks(row_numbers, data)
    filtered_docs.extend(additional_docs)

    # print(filtered_docs)

    return filtered_docs


def initialize_rag(llm, data, retriever, chat_history):
    def wrapped_retriever(input_data):

        input_query = input_data.content
        # print("chat history:", chat_history)

        contextualized_query = contextualize_query(input_query, chat_history)
        # print("New contextualized query:", contextualized_query)

        return retrieve_and_combine_documents(contextualized_query.content, chat_history, data, retriever)
        # chat_history is still passed, and it's from the from_template

    def contextualize_query(query, history):
        # Use the ChatPromptTemplate to reformulate the query
        # This way seems far-fetched, but let's overlook for better good

        prompt = contextualize_q_prompt.format(input=query, chat_history=history)
        return llm.invoke(prompt)

    # contextualize_q_system_prompt = """Given a chat history and the latest user question \
    # which might reference context in the chat history, formulate a standalone question \
    # which can be understood without the chat history. Do NOT answer the question, \
    # just reformulate it if needed and otherwise return it as is.
    # """

    information_replacement = """
    <information> means you have to fill in the appropriate information based on the context of the conversation.
    Eg: We have this retrieved data: loa pasion 10, giá: 10VND, công suất: 100W.
    Answer form: Vâng, chúng tôi có bán Loa JBL Pasion 10 với <các thông tin cần thiết>
    ==> Actual answer: Vâng, chúng tôi có bán Loa JBL Pasion 10 với giá là 10VND và công suất là 100W.
    """

    feedback_content = """
      Here is the feedback of customers. Please learn from this feedback so that you don't repeat your mistakes.
      Learn the correct format after "as the feedback is" so that you can apply the format for other similar questions.
      <information> means you have to fill in the appropriate information based on the context of the conversation.
      You don't have to use the exact content in Correction value, just fill in the appropriate information, unless it requires correct format.
    """

    contextualize_q_system_prompt = """
    Bạn là 1 nhà tư vấn thiết bị âm thanh, gồm loa, micro, mixer, ampli,...

    Dựa trên lịch sử trò chuyện dưới đây và câu hỏi mới nhất của khách hàng, hãy diễn giải câu hỏi sao cho dễ hiểu và liên quan đến ngữ cảnh đã trao đổi.

    1. Sử dụng thông tin từ lịch sử để thêm chi tiết còn thiếu cho câu hỏi mới (nếu có).
    2. Đảm bảo rằng câu hỏi được diễn giải một cách chính xác và ngắn gọn, nhưng vẫn giữ ngữ cảnh từ lịch sử trò chuyện.
    3. Nếu không có đủ thông tin từ lịch sử, hãy diễn giải câu hỏi mới sao cho dễ hiểu nhất mà không cần ngữ cảnh.

    Ví dụ:
    - Lịch sử: "Cho tôi biết giá loa JBL 201 Seri 4"
    - Câu mới: "Gửi tôi link sản phẩm"
    - Diễn giải: "Bạn có thể cho tôi biết link sản phẩm của loa JBL 201 Seri 4 không?"

    4. Nếu câu trả lời của bạn đề cập đến một sản phẩm, bạn PHẢI kèm theo đường link sản phẩm trong dữ liệu được cung cấp.

    Ví dụ:
    - Câu hỏi: "Cho tôi biết giá loa JBL 201 Seri 4"
    - Trả lời: "Dạ, hiện tại, giá của loa là $$$:
    [<sản phẩm được đề cập>](<link sản phẩm)"

    Lịch sử trò chuyện:
    {chat_history}

    Câu hỏi mới: {input}

    Diễn giải:
    """

    # Accumulate corrections based on feedback dataframe
    for index, row in feedback_df.iterrows():
        if pd.notna(row['Correction']):
            feedback_content += f"""
            If a user asks: \"{row['Query']}\", you shouldn't answer like this: \"{row['Response']}\",
            as the feedback is {row['Feedback']}, but you should answer: {row['Correction']}\n\n
            """

    input_premise = """
    - Không được trả lời lại câu yêu cầu, chỉ diễn giải lại nó sao cho phù hợp với những gì được fine-tuned.
    - Bắt buộc sử dụng tiếng Việt nếu khách hàng không yêu cầu ngôn ngữ khác.
    - When customers ask for product suggestions, paraphrase the customer's question to include discount keywords, to prioritize finding products with discounts. If there are no suitable promotional products, then choose the remaining products.    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             contextualize_q_system_prompt + "\n\n"
            #  + contextualized_prompt +  "\n\n"
             + input_premise +  "\n\n"
             + information_replacement
             ), # premise +  "\n\n" + feedback_content + "\n\n" + "\n\n" +  keyword_feedback
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a history-aware retriever using the custom wrapped retriever
    history_aware_retriever = contextualize_q_prompt | llm | wrapped_retriever

    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", premise +  "\n\n" + information_replacement + "\n\n" + feedback_content +  "\n\n" + "{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    # Initialize memory and QA system
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create and return the RAG chain
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Example usage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.1)
retriever = docsearch.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.4,
        "k": 20
        }
    )

from langsmith import Client

client = Client()

# csv_path = "Finished_Data_in_769audio_vn.csv"
csv_path = "Final_Cleaned_Dataset.xlsx - Dataset.csv"

data = pd.read_csv(csv_path, encoding="utf-8")
# print(feedback_df.head())

# from langchain_core.messages import HumanMessage, AIMessage
#
# query = "201 seri 4 nhieu vay chi"
# chat_history = [
#     HumanMessage(content="201 seri 4 nhieu vay chi"),
#     # AIMessage(content="Dạ, hiện tại bên chúng tôi có bán Loa Bose 201 seri IV với giá là 4,200,000 VNĐ."),
#     # HumanMessage(content=query)
# ]
# chat_history.extend([HumanMessage(content=query)])
#
# rag = initialize_rag(llm, data, retriever)
#
# from langchain import callbacks
#
# with callbacks.collect_runs() as cb:
#   result = rag.invoke({"input": query, "chat_history": chat_history})
#   run_id = cb.traced_runs[0].id
#
# print(result['answer'])
# print(run_id)
