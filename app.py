import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
import sqlite3
from datetime import datetime
import os
from dotenv import load_dotenv

# Database initialization
def init_sqlite_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    
    # Create threads table
    c.execute('''
        CREATE TABLE IF NOT EXISTS threads (
            thread_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create messages table
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id INTEGER,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (thread_id) REFERENCES threads (thread_id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Thread operations
def create_thread(title):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('INSERT INTO threads (title) VALUES (?)', (title,))
    thread_id = c.lastrowid
    conn.commit()
    conn.close()
    return thread_id

def get_all_threads():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('SELECT thread_id, title, created_at FROM threads ORDER BY created_at DESC')
    threads = c.fetchall()
    conn.close()
    return threads

def get_thread_messages(thread_id):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('SELECT role, content FROM messages WHERE thread_id = ? ORDER BY timestamp', (thread_id,))
    messages = [{"role": role, "content": content} for role, content in c.fetchall()]
    conn.close()
    return messages

def save_message(thread_id, role, content):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('INSERT INTO messages (thread_id, role, content) VALUES (?, ?, ?)',
              (thread_id, role, content))
    conn.commit()
    conn.close()

# Initialize DB connection and LangChain components (your existing code)
load_dotenv()
DB_URI = os.getenv("DB_URI")
db = SQLDatabase.from_uri(DB_URI)

def get_schema(_):
    return db.get_table_info()

llm = ChatOpenAI(streaming=True)

# Your existing LangChain chains (template, prompt, sql_chain, etc.)
template = """
Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
Based on the table schema below, write an SQL query that would answer the user's question.
{schema}

Question: {question}
SQL Query:"""

prompt = ChatPromptTemplate.from_template(template)

sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm.bind(stop="\nSQL Result:")
    | StrOutputParser()
)

template_response = """
Based on the table schema below, sql query, and sql response, write a natural language response and display the query that was ran as a block of code:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""

prompt_response = ChatPromptTemplate.from_template(template_response)

def run_query(query):
    try:
        return db.run(query)
    except Exception as e:
        error_message = str(e)
        print(f"Error executing query: {query}\nError: {error_message}")
        if "syntax error" in error_message.lower():
            return "There was a syntax error in the generated SQL query. Please try rephrasing your question."
        else:
            return f"An error occurred while executing the query: {error_message}"

full_chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
        schema=get_schema,
        response=lambda vars: run_query(vars["query"]),
    )
    | prompt_response
    | llm
    | StrOutputParser()
)

def get_thread_title(thread_id):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('SELECT title FROM threads WHERE thread_id = ?', (thread_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

# Streamlit UI
def main():
    st.set_page_config(page_title="IIF Database Chat", page_icon="🤖", layout="wide")
    
    # Initialize SQLite database
    init_sqlite_db()
    
    # st.title("IIF Database Chat Interface")

    # Current thread display
    if "current_thread_id" in st.session_state and st.session_state.current_thread_id is not None:
        thread_title = get_thread_title(st.session_state.current_thread_id)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### 📝 Current Thread: *{thread_title}*")
        # with col2:
        #     if st.button("Start New Thread", type="secondary"):
        #         st.session_state.current_thread_id = None
        #         st.session_state.messages = []
        #         st.rerun()
    
    # Sidebar with thread management
    with st.sidebar:
        st.title("IIF SQL Database Chat")
        st.divider()
        st.header("Chat Threads")
        
        # Create new thread
        new_thread_title = st.text_input("New Thread Title")
        create_thread_button = st.button("Create New Thread", type='primary', disabled=not new_thread_title)

        if create_thread_button:
            if new_thread_title:
                thread_id = create_thread(new_thread_title)
                st.session_state.current_thread_id = thread_id
                st.session_state.messages = []
                st.rerun()
        
        # List and select existing threads
        st.subheader("Existing Threads")
        threads = get_all_threads()
        for thread_id, title, created_at in threads:
            if st.button(f"{title} ({created_at})", key=f"thread_{thread_id}"):
                st.session_state.current_thread_id = thread_id
                st.session_state.messages = get_thread_messages(thread_id)
                st.rerun()
        
        # Database schema information
        st.header("Database Information")
        st.write("Schema:")
        st.code(get_schema(None))
    
    # Initialize session state
    if "current_thread_id" not in st.session_state:
        st.session_state.current_thread_id = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Main chat interface
    if st.session_state.current_thread_id is None:
        st.info("Please create a new thread or select an existing one to start chatting.")
    else:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your database"):
            # Display and save user message
            with st.chat_message("user"):
                st.markdown(prompt)
            save_message(st.session_state.current_thread_id, "user", prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                try:
                    with st.spinner("Thinking..."):
                        response = full_chain.invoke({"question": prompt})
                        response_placeholder.markdown(response)
                        save_message(st.session_state.current_thread_id, "assistant", response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    response_placeholder.error(error_message)
                    save_message(st.session_state.current_thread_id, "assistant", error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()