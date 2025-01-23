# from langchain.vectorstores import FAISS
# from langchain.embeddings import OllamaEmbeddings
# from langchain.llms import Ollama
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import streamlit as st

data = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EDS UI Library Example</title>
  <style>
    /* Buttons */
    .eds-btn {
      padding: 10px 20px;
      font-size: 16px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      display: inline-block;
      background:red
    }

    .eds-btn-primary {
      background-color: #007bff;
      color: white;
    }

    .eds-btn-secondary {
      background-color: #6c757d;
      color: white;
    }

    /* Containers */
    .eds-container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 15px;
    }

    .eds-container-fluid {
      width: 100%;
      padding: 15px;
    }

    /* Forms */
    .eds-form {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .eds-input {
      width: 100%;
      padding: 8px;
      font-size: 14px;
      border: 1px solid #ccc;
      border-radius: 4px;
    }

    .eds-input-lg {
      padding: 12px;
      font-size: 16px;
    }

    .eds-input-sm {
      padding: 6px;
      font-size: 12px;
    }

    .eds-input-disabled {
      background-color: #e9ecef;
      cursor: not-allowed;
    }

    /* Tables */
    .eds-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }

    .eds-table th,
    .eds-table td {
      border: 1px solid #ddd;
      padding: 8px;
    }

    .eds-table-striped tbody tr:nth-child(odd) {
      background-color: #f9f9f9;
    }

    .eds-table-bordered {
      border: 2px solid #ddd;
    }

    .eds-table-hover tbody tr:hover {
      background-color: #f1f1f1;
    }

    .eds-table-sm th,
    .eds-table-sm td {
      padding: 4px;
    }

    /* Cards */
    .eds-card {
      border: 1px solid #ddd;
      border-radius: 4px;
      overflow: hidden;
      margin-bottom: 10px;
    }

    .eds-card-header,
    .eds-card-footer {
      background-color: #f7f7f7;
      padding: 10px;
      font-weight: bold;
    }

    .eds-card-body {
      padding: 10px;
    }

    /* Modals */
    .eds-modal {
      background-color: white;
      border: 1px solid #ddd;
      border-radius: 4px;
      padding: 20px;
      max-width: 500px;
      margin: 50px auto;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .eds-modal-header,
    .eds-modal-footer {
      font-weight: bold;
      padding: 10px 0;
    }

    .eds-modal-body {
      padding: 10px 0;
    }

    /* Navigation */
    .eds-nav {
      display: flex;
      gap: 10px;
      background-color: #f8f9fa;
      padding: 10px;
    }

    .eds-nav-item {
      list-style: none;
    }

    .eds-nav-link {
      color: #007bff;
      text-decoration: none;
      padding: 5px 10px;
    }

    .eds-nav-link:hover {
      background-color: #e9ecef;
      border-radius: 4px;
    }

    /* Alerts */
    .eds-alert {
      padding: 15px;
      border-radius: 4px;
      margin-bottom: 10px;
    }

    .eds-alert-success {
      background-color: #d4edda;
      color: #155724;
    }

    .eds-alert-danger {
      background-color: #f8d7da;
      color: #721c24;
    }

    /* Pagination */
    .eds-pagination {
      display: flex;
      list-style: none;
      gap: 5px;
      padding: 0;
    }

    .eds-pagination-item {
      display: inline-block;
    }

    .eds-pagination-link {
      display: block;
      padding: 8px 12px;
      border: 1px solid #ddd;
      color: #007bff;
      text-decoration: none;
    }

    .eds-pagination-link:hover {
      background-color: #e9ecef;
    }
  </style>
</head>
<body>
  <!-- Buttons -->
  <button class="eds-btn eds-btn-primary">Primary Button</button>
  <button class="eds-btn eds-btn-secondary">Secondary Button</button>

  <!-- Containers -->
  <div class="eds-container">Fixed-width content goes here.</div>
  <div class="eds-container-fluid">Full-width content goes here.</div>

  <!-- Forms -->
  <div class="eds-form">
    <label for="username">Username</label>
    <input type="text" id="username" class="eds-input" placeholder="Enter your username">
    <label for="password">Password</label>
    <input type="password" id="password" class="eds-input" placeholder="Enter your password">
    <button class="eds-btn eds-btn-primary">Login</button>
  </div>

  <!-- Tables -->
  <table class="eds-table eds-table-striped eds-table-bordered">
    <thead>
      <tr>
        <th>Name</th>
        <th>Age</th>
        <th>Country</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>John Doe</td>
        <td>28</td>
        <td>USA</td>
      </tr>
      <tr>
        <td>Jane Smith</td>
        <td>32</td>
        <td>Canada</td>
      </tr>
    </tbody>
  </table>

  <!-- Cards -->
  <div class="eds-card">
    <div class="eds-card-header">Card Header</div>
    <div class="eds-card-body">This is the body of the card.</div>
    <div class="eds-card-footer">Card Footer</div>
  </div>

  <!-- Modals -->
  <div class="eds-modal">
    <div class="eds-modal-header">Modal Header</div>
    <div class="eds-modal-body">This is the modal body.</div>
    <div class="eds-modal-footer">
      <button class="eds-btn eds-btn-secondary">Close</button>
    </div>
  </div>

  <!-- Navigation -->
  <nav class="eds-nav">
    <a class="eds-nav-link eds-nav-item" href="#">Home</a>
    <a class="eds-nav-link eds-nav-item" href="#">About</a>
    <a class="eds-nav-link eds-nav-item" href="#">Services</a>
    <a class="eds-nav-link eds-nav-item" href="#">Contact</a>
  </nav>

  <!-- Alerts -->
  <div class="eds-alert eds-alert-success">This is a success alert!</div>
  <div class="eds-alert eds-alert-danger">This is a danger alert!</div>

  <!-- Pagination -->
  <ul class="eds-pagination">
    <li class="eds-pagination-item">
      <a href="#" class="eds-pagination-link">&laquo;</a>
    </li>
    <li class="eds-pagination-item">
      <a href="#" class="eds-pagination-link">1</a>
    </li>
    <li class="eds-pagination-item">
      <a href="#" class="eds-pagination-link">2</a>
    </li>
    <li class="eds-pagination-item">
      <a href="#" class="eds-pagination-link">&raquo;</a>
    </li>
  </ul>
</body>
</html>

"""

# Simulate loading the data into a document format (as if it's from a file)
documents = [ {"page_content":data}]

# # Split the text into smaller chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# docs = text_splitter.split_documents(documents)

# # Initialize the embedding function and FAISS vector database
# embedding = OllamaEmbeddings(model="gemma:2b")
# vectordb = FAISS.from_documents(docs, embedding)

# # Define the Ollama model and prompt
# llm = Ollama(model="gemma:2b")
# template = """Use the following pieces of context to answer the question at the end.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Use three sentences maximum. Always say "Thanks for asking!" at the end.
# {context}
# Question: {question}
# Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

# # Create a RetrievalQA chain
# qa_chain = RetrievalQA.from_chain_type(llm,
#                                        retriever=vectordb.as_retriever(),
#                                        return_source_documents=True,
#                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

# # Streamlit UI
# st.title("Document Q&A with Ollama gemmaL2b and FAISS")
# st.write("This app uses a preloaded text file to answer your questions. Ask away!")

# # User Input for Question
# question = st.text_input("Ask a question about the document:")
# if question:
#     with st.spinner("Thinking..."):
#         # Run the chain to get the answer
#         result = qa_chain({"query": question})
    
#     # Display results
#     st.write("### Answer:")
#     st.write(result["result"])
    
#     # Optionally display source documents
#     with st.expander("View Source Documents"):
#         for i, doc in enumerate(result["source_documents"]):
#             st.write(f"#### Document {i + 1}:")
#             st.write(doc.page_content)

from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embedding = OllamaEmbeddings(model="gemma:2b")
vectordb = FAISS.from_documents(docs, embedding)

llm = Ollama(model="gemma:2b")

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""
    You are an AI assistant trained to answer questions about the EDS UI Library.
    Here is the context:
    {context}
    
    Chat History:
    {chat_history}
    
    Question:
    {question}
    
    Provide a concise and informative answer.
    """
)

class ChatHistoryQAChain:
    def __init__(self, llm, retriever, prompt_template):
        self.llm = llm
        self.retriever = retriever
        self.prompt_template = prompt_template

    def __call__(self, inputs):
        inputs["chat_history"] = inputs.get("chat_history", "")
        context_docs = self.retriever.get_relevant_documents(inputs["query"])
        context = "\n".join([doc.page_content for doc in context_docs])
        prompt = self.prompt_template.format(
            context=context,
            chat_history=inputs["chat_history"],
            question=inputs["query"],
        )
        response = self.llm(prompt)
        return {"result": response, "source_documents": context_docs}

qa_chain = ChatHistoryQAChain(
    llm=llm, retriever=vectordb.as_retriever(), prompt_template=QA_CHAIN_PROMPT
)

st.title("EDS-Aware GenAI App")
st.write("Get tailored responses for your company's EDS UI Library.")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = ""

question = st.text_input("Ask about the EDS UI Library:")

if question:
    st.session_state["chat_history"] += f"User: {question}\n"
    with st.spinner("Processing..."):
        result = qa_chain({"query": question, "chat_history": st.session_state["chat_history"]})
        answer = result["result"]

    st.session_state["chat_history"] += f"AI: {answer}\n"
    st.write("### Answer:")
    st.write(answer)

    with st.expander("View Source Documents"):
        for i, doc in enumerate(result["source_documents"]):
            st.write(f"Document {i + 1}: {doc.page_content}")

    with st.expander("View Chat History"):
        st.text(st.session_state["chat_history"])
