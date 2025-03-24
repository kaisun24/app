import gradio as gr
import groq
import os
import tempfile
import uuid
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
import base64
from PIL import Image
import io
import requests
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
import torch
import numpy as np

# Load environment variables
load_dotenv()
client = groq.Client(api_key=os.getenv("GROQ_TECH_API_KEY"))

# Initialize embeddings with error handling
try:
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
except Exception as e:
    print(f"Error loading embeddings: {e}")
    embeddings = None

# Directory to store FAISS indexes with better naming
FAISS_INDEX_DIR = "faiss_indexes_tech_cpu"
if not os.path.exists(FAISS_INDEX_DIR):
    os.makedirs(FAISS_INDEX_DIR)

# Dictionary to store user-specific vectorstores
user_vectorstores = {}

# Custom CSS to match HTML exactly
custom_css = """
:root {
    --primary-color: #4285F4;
    --secondary-color: #34A853;
    --accent-color: #EA4335;
    --yellow-color: #FBBC05;
    --light-background: #F8F9FA;
    --dark-text: #202124;
    --white: #FFFFFF;
    --border-color: #DADCE0;
    --code-bg: #F1F3F4;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    --shadow-md: 0 4px 6px rgba(0,0,0,0.1);
    --shadow-lg: 0 10px 20px rgba(0,0,0,0.1);
    --transition: all 0.3s cubic-bezier(.25,.8,.25,1);
}

/* Global Styles */
body {
    background-color: var(--light-background);
    font-family: 'Roboto', sans-serif;
    color: var(--dark-text);
    line-height: 1.6;
}

/* Header */
.header {
    background-color: var(--white);
    box-shadow: var(--shadow-sm);
    position: sticky;
    top: 0;
    z-index: 100;
    padding: 16px 0;
}

.logo {
    display: flex;
    align-items: center;
    gap: 12px;
}

.logo-icon {
    color: var(--primary-color);
    font-size: 24px;
}

/* Main Layout */
.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 40px 20px;
}

.main-content {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 30px;
}

/* Sidebar */
.sidebar {
    background-color: var(--white);
    border-radius: 12px;
    box-shadow: var(--shadow-md);
    overflow: hidden;
}

.sidebar-section {
    padding: 20px;
    border-bottom: 1px solid var(--border-color);
}

/* File Upload */
.file-upload {
    border: 2px dashed var(--border-color);
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    cursor: pointer;
    transition: var(--transition);
    margin-bottom: 16px;
}

/* Tabs */
.tabs {
    display: flex;
    background-color: var(--white);
    border-radius: 12px 12px 0 0;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}

.tab {
    padding: 16px 24px;
    font-family: 'Google Sans', sans-serif;
    font-weight: 500;
    cursor: pointer;
    transition: var(--transition);
    border-bottom: 3px solid transparent;
}

.tab.active {
    color: var(--primary-color);
    border-bottom-color: var(--primary-color);
}

/* Chat Section */
.chat-section {
    background-color: var(--white);
    border-radius: 12px;
    box-shadow: var(--shadow-md);
    overflow: hidden;
    margin-top: 24px;
}

.chat-header {
    padding: 16px;
    background-color: var(--primary-color);
    color: var(--white);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.chat-messages {
    height: 400px;
    overflow-y: auto;
    padding: 16px;
}

.message {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: 18px;
    margin-bottom: 12px;
}

.message-user {
    background-color: var(--primary-color);
    color: var(--white);
    margin-left: auto;
    border-radius: 18px 18px 4px 18px;
}

.message-bot {
    background-color: var(--light-background);
    color: var(--dark-text);
    margin-right: auto;
    border-radius: 18px 18px 18px 4px;
}

/* Buttons */
.primary-button {
    background-color: var(--primary-color);
    color: var(--white);
    border: none;
    border-radius: 4px;
    padding: 10px 16px;
    font-family: 'Google Sans', sans-serif;
    font-weight: 500;
    cursor: pointer;
    transition: var(--transition);
}

.primary-button:hover {
    background-color: #3367d6;
}
"""

# Custom JavaScript to enhance UI
custom_js = """
document.addEventListener('DOMContentLoaded', function() {
    // Add Font Awesome
    const fontAwesome = document.createElement('link');
    fontAwesome.rel = 'stylesheet';
    fontAwesome.href = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css';
    document.head.appendChild(fontAwesome);
    
    // Add Google Fonts
    const googleFonts = document.createElement('link');
    googleFonts.rel = 'stylesheet';
    googleFonts.href = 'https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@300;400;500&family=Roboto+Mono&display=swap';
    document.head.appendChild(googleFonts);
    
    // Initialize UI enhancements
    setTimeout(enhanceUI, 1000);
});

function enhanceUI() {
    // Add icons to headers
    addIconToHeader('Upload Code', 'fa-upload');
    addIconToHeader('Developer Tools', 'fa-tools');
    addIconToHeader('Tech Assistant', 'fa-robot');
    
    // Setup tabs
    setupTabs();
    
    // Setup file upload area
    setupFileUpload();
}

function addIconToHeader(text, iconClass) {
    document.querySelectorAll('h3').forEach(header => {
        if (header.textContent.includes(text)) {
            const icon = document.createElement('i');
            icon.className = `fas ${iconClass}`;
            header.insertBefore(icon, header.firstChild);
            header.style.display = 'flex';
            header.style.alignItems = 'center';
            header.style.gap = '8px';
        }
    });
}

function setupTabs() {
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
        });
    });
}

function setupFileUpload() {
    const dropzone = document.querySelector('.file-upload');
    if (!dropzone) return;
    
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--primary-color)';
        dropzone.style.backgroundColor = 'rgba(66, 133, 244, 0.05)';
    });
    
    dropzone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--border-color)';
        dropzone.style.backgroundColor = 'transparent';
    });
}
"""

# Helper functions for code analysis
def detect_language(extension):
    """Detect programming language from file extension"""
    extension_map = {
        ".py": "Python",
        ".js": "JavaScript",
        ".java": "Java",
        ".cpp": "C++",
        ".c": "C",
        ".cs": "C#",
        ".php": "PHP",
        ".rb": "Ruby",
        ".go": "Go",
        ".ts": "TypeScript"
    }
    return extension_map.get(extension.lower(), "Unknown")

def calculate_complexity_metrics(content, language):
    """Calculate code complexity metrics"""
    lines = content.split('\n')
    total_lines = len(lines)
    blank_lines = len([line for line in lines if not line.strip()])
    code_lines = total_lines - blank_lines
    
    metrics = {
        "language": language,
        "total_lines": total_lines,
        "code_lines": code_lines,
        "blank_lines": blank_lines
    }
    
    return metrics

def generate_recommendations(metrics):
    """Generate code quality recommendations based on metrics"""
    recommendations = []
    
    if metrics.get("cyclomatic_complexity", 0) > 10:
        recommendations.append("🔄 High cyclomatic complexity detected. Consider breaking down complex functions.")
    
    if metrics.get("code_lines", 0) > 300:
        recommendations.append("📏 File is quite large. Consider splitting it into multiple modules.")
    
    if metrics.get("functions", 0) > 10:
        recommendations.append("🔧 Large number of functions. Consider grouping related functions into classes.")
    
    if metrics.get("comments", 0) / max(metrics.get("code_lines", 1), 1) < 0.1:
        recommendations.append("📝 Low comment ratio. Consider adding more documentation.")
    
    return "### Recommendations\n\n" + "\n\n".join(recommendations) if recommendations else ""

# Function to process PDF files
def process_pdf(pdf_file):
    if pdf_file is None:
        return None, "No file uploaded", {"page_images": [], "total_pages": 0, "total_words": 0}
    try:
        session_id = str(uuid.uuid4())
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_file)
            pdf_path = temp_file.name
        
        doc = fitz.open(pdf_path)
        texts = [page.get_text() for page in doc]
        page_images = []
        for page in doc:
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            page_images.append(img_base64)
        total_pages = len(doc)
        total_words = sum(len(text.split()) for text in texts)
        doc.close()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.create_documents(texts)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        index_path = os.path.join(FAISS_INDEX_DIR, session_id)
        vectorstore.save_local(index_path)
        user_vectorstores[session_id] = vectorstore

        os.unlink(pdf_path)
        pdf_state = {"page_images": page_images, "total_pages": total_pages, "total_words": total_words}
        return session_id, f"✅ Successfully processed {len(chunks)} text chunks from your PDF", pdf_state
    except Exception as e:
        if "pdf_path" in locals() and os.path.exists(pdf_path):
            os.unlink(pdf_path)
        return None, f"Error processing PDF: {str(e)}", {"page_images": [], "total_pages": 0, "total_words": 0}

# Function to generate chatbot responses with Tech theme
def generate_response(message, session_id, model_name, history):
    """Generate chatbot responses with FAISS context enhancement"""
    if not message:
        return history
    
    try:
        context = ""
        if embeddings and session_id and session_id in user_vectorstores:
            try:
                print(f"Performing similarity search with session: {session_id}")
                vectorstore = user_vectorstores[session_id]
                
                # Use a higher k value to get more relevant context
                docs = vectorstore.similarity_search(message, k=5)
                
                if docs:
                    # Format the context more clearly with source information
                    context = "\n\nRelevant code context from your files:\n\n"
                    for i, doc in enumerate(docs, 1):
                        source = doc.metadata.get("source", "Unknown")
                        language = doc.metadata.get("language", "Unknown")
                        context += f"--- Segment {i} from {source} ({language}) ---\n"
                        context += f"```\n{doc.page_content}\n```\n\n"
                    
                    print(f"Found {len(docs)} relevant code segments for context.")
            except Exception as e:
                print(f"Warning: Failed to perform similarity search: {e}")
        
        system_prompt = """You are a technical assistant specializing in software development and programming.
        Provide clear, accurate responses with code examples when relevant.
        Format code snippets with proper markdown code blocks and specify the language."""
        
        if context:
            system_prompt += f"\n\nUse this context from the uploaded code files to inform your answers:{context}"
        
        # Add instruction to reference specific file parts
        system_prompt += "\nWhen discussing code from the uploaded files, specifically reference the file name and segment number."
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        
        response = completion.choices[0].message.content
        
        # For proper chat history handling
        if isinstance(history, list) and history and isinstance(history[0], dict):
            # History is in message format
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
        else:
            # Fallback for other formats
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
        
        return history
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        
        # Handle different history formats
        if isinstance(history, list):
            if history and isinstance(history[0], dict):
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": error_msg})
            else:
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": error_msg})
        
        return history

# Functions to update PDF viewer
def update_pdf_viewer(pdf_state):
    if not pdf_state["total_pages"]:
        return 0, None, "No PDF uploaded yet"
    try:
        img_data = base64.b64decode(pdf_state["page_images"][0])
        img = Image.open(io.BytesIO(img_data))
        return pdf_state["total_pages"], img, f"**Total Pages:** {pdf_state['total_pages']}\n**Total Words:** {pdf_state['total_words']}"
    except Exception as e:
        print(f"Error decoding image: {e}")
        return 0, None, "Error displaying PDF"

def update_image(page_num, pdf_state):
    if not pdf_state["total_pages"] or page_num < 1 or page_num > pdf_state["total_pages"]:
        return None
    try:
        img_data = base64.b64decode(pdf_state["page_images"][page_num - 1])
        img = Image.open(io.BytesIO(img_data))
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

# GitHub API integration
def search_github_repos(query, sort="stars", order="desc", per_page=10):
    """Search for GitHub repositories"""
    try:
        github_token = os.getenv("GITHUB_TOKEN", "")
        headers = {}
        if github_token:
            headers["Authorization"] = f"token {github_token}"
            
        params = {
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": per_page
        }
        
        response = requests.get(
            "https://api.github.com/search/repositories",
            headers=headers,
            params=params
        )
        
        if response.status_code != 200:
            print(f"GitHub API Error: {response.status_code} - {response.text}")
            return []
            
        data = response.json()
        return data.get("items", [])
    except Exception as e:
        print(f"Error in GitHub search: {e}")
        return []

# Stack Overflow API integration
def search_stackoverflow(query, sort="votes", site="stackoverflow", pagesize=10):
    """Search for questions on Stack Overflow"""
    try:
        params = {
            "order": "desc",
            "sort": sort,
            "site": site,
            "pagesize": pagesize,
            "intitle": query
        }
        
        response = requests.get(
            "https://api.stackexchange.com/2.3/search/advanced",
            params=params
        )
        
        if response.status_code != 200:
            print(f"Stack Exchange API Error: {response.status_code} - {response.text}")
            return []
            
        data = response.json()
        
        # Process results to convert Unix timestamps to readable dates
        for item in data.get("items", []):
            if "creation_date" in item:
                item["creation_date"] = datetime.fromtimestamp(item["creation_date"]).strftime("%Y-%m-%d")
                
        return data.get("items", [])
    except Exception as e:
        print(f"Error in Stack Overflow search: {e}")
        return []

def get_stackoverflow_answers(question_id, site="stackoverflow"):
    """Get answers for a specific question on Stack Overflow"""
    try:
        params = {
            "order": "desc",
            "sort": "votes",
            "site": site,
            "filter": "withbody"  # Include the answer body in the response
        }
        
        response = requests.get(
            f"https://api.stackexchange.com/2.3/questions/{question_id}/answers",
            params=params
        )
        
        if response.status_code != 200:
            print(f"Stack Exchange API Error: {response.status_code} - {response.text}")
            return []
            
        data = response.json()
        
        # Process results
        for item in data.get("items", []):
            if "creation_date" in item:
                item["creation_date"] = datetime.fromtimestamp(item["creation_date"]).strftime("%Y-%m-%d")
                
        return data.get("items", [])
    except Exception as e:
        print(f"Error getting Stack Overflow answers: {e}")
        return []

def explain_code(code):
    """Explain code using LLM"""
    try:
        system_prompt = "You are an expert programmer and code reviewer. Your task is to explain the provided code in a clear, concise manner. Include:"
        system_prompt += "\n1. What the code does (high-level overview)"
        system_prompt += "\n2. Key functions/components and their purposes"
        system_prompt += "\n3. Potential issues or optimization opportunities"
        system_prompt += "\n4. Any best practices that are followed or violated"
        
        completion = client.chat.completions.create(
            model="llama3-70b-8192",  # Using more capable model for code explanation
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Explain this code:\n```\n{code}\n```"}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        
        explanation = completion.choices[0].message.content
        return f"**Code Explanation:**\n\n{explanation}"
    except Exception as e:
        return f"Error explaining code: {str(e)}"

def perform_repo_search(query, language, sort_by, min_stars):
    """Perform GitHub repository search with UI parameters"""
    try:
        if not query:
            return "Please enter a search query"
            
        # Build the search query with filters
        search_query = query
        if language and language != "any":
            search_query += f" language:{language}"
        if min_stars and min_stars != "0":
            search_query += f" stars:>={min_stars}"
            
        # Map sort_by to GitHub API parameters
        sort_param = "stars"
        if sort_by == "updated":
            sort_param = "updated"
        elif sort_by == "forks":
            sort_param = "forks"
            
        results = search_github_repos(search_query, sort=sort_param)
        
        if not results:
            return "No repositories found. Try different search terms."
            
        # Format results as markdown
        markdown = "## GitHub Repository Search Results\n\n"
        
        for i, repo in enumerate(results, 1):
            markdown += f"### {i}. [{repo['full_name']}]({repo['html_url']})\n\n"
            
            if repo['description']:
                markdown += f"{repo['description']}\n\n"
                
            markdown += f"**Language:** {repo['language'] or 'Not specified'}\n"
            markdown += f"**Stars:** {repo['stargazers_count']} | **Forks:** {repo['forks_count']} | **Watchers:** {repo['watchers_count']}\n"
            markdown += f"**Created:** {repo['created_at'][:10]} | **Updated:** {repo['updated_at'][:10]}\n\n"
            
            if repo.get('topics'):
                markdown += f"**Topics:** {', '.join(repo['topics'])}\n\n"
                
            if repo.get('license') and repo['license'].get('name'):
                markdown += f"**License:** {repo['license']['name']}\n\n"
                
            markdown += f"[View Repository]({repo['html_url']}) | [Clone URL]({repo['clone_url']})\n\n"
            markdown += "---\n\n"
            
        return markdown
    except Exception as e:
        return f"Error searching for repositories: {str(e)}"

def perform_stack_search(query, tag, sort_by):
    """Perform Stack Overflow search with UI parameters"""
    try:
        if not query:
            return "Please enter a search query"
            
        # Add tag to query if specified
        if tag and tag != "any":
            query_with_tag = f"{query} [tag:{tag}]"
        else:
            query_with_tag = query
            
        # Map sort_by to Stack Exchange API parameters
        sort_param = "votes"
        if sort_by == "newest":
            sort_param = "creation"
        elif sort_by == "activity":
            sort_param = "activity"
            
        results = search_stackoverflow(query_with_tag, sort=sort_param)
        
        if not results:
            return "No questions found. Try different search terms."
            
        # Format results as markdown
        markdown = "## Stack Overflow Search Results\n\n"
        
        for i, question in enumerate(results, 1):
            markdown += f"### {i}. [{question['title']}]({question['link']})\n\n"
            
            # Score and answer stats
            markdown += f"**Score:** {question['score']} | **Answers:** {question['answer_count']}"
            if question.get('is_answered'):
                markdown += " ✓ (Accepted answer available)"
            markdown += "\n\n"
            
            # Tags
            if question.get('tags'):
                markdown += "**Tags:** "
                for tag in question['tags']:
                    markdown += f"`{tag}` "
                markdown += "\n\n"
                
            # Asked info
            markdown += f"**Asked:** {question['creation_date']} | **Views:** {question.get('view_count', 'N/A')}\n\n"
            
            markdown += f"[View Question]({question['link']})\n\n"
            markdown += "---\n\n"
            
        return markdown
    except Exception as e:
        return f"Error searching Stack Overflow: {str(e)}"

# Modify the process_code_file function
def process_code_file(file_obj):
    """Process uploaded code files and store in FAISS index"""
    if file_obj is None:
        return None, "No file uploaded", {}
    
    try:
        # Handle both file objects and bytes objects
        if isinstance(file_obj, bytes):
            content = file_obj.decode('utf-8', errors='replace')  # Added error handling
            file_name = "uploaded_file"
            file_extension = ".txt"  # Default extension
        else:
            content = file_obj.read().decode('utf-8', errors='replace')  # Added error handling
            file_name = getattr(file_obj, 'name', 'uploaded_file')
            file_extension = Path(file_name).suffix.lower()
            
        language = detect_language(file_extension)
        
        # Calculate metrics
        metrics = calculate_complexity_metrics(content, language)
        
        # Create vectorstore if embeddings are available
        session_id = None
        if embeddings:
            try:
                print(f"Creating FAISS index for {file_name}...")
                # Improved chunking for code files
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,  # Smaller chunks for code
                    chunk_overlap=50,
                    separators=["\n\n", "\n", " ", ""]
                )
                chunks = text_splitter.create_documents([content], metadatas=[{"filename": file_name, "language": language}])
                
                # Add source metadata to help with retrieval
                for i, chunk in enumerate(chunks):
                    chunk.metadata["chunk_id"] = i
                    chunk.metadata["source"] = file_name
                
                # Create and store vectorstore
                vectorstore = FAISS.from_documents(chunks, embeddings)
                session_id = str(uuid.uuid4())
                index_path = os.path.join(FAISS_INDEX_DIR, session_id)
                vectorstore.save_local(index_path)
                user_vectorstores[session_id] = vectorstore
                
                # Add number of chunks to metrics for display
                metrics["chunks"] = len(chunks)
                print(f"Successfully created FAISS index with {len(chunks)} chunks.")
            except Exception as e:
                print(f"Warning: Failed to create vectorstore: {e}")
        
        return session_id, f"✅ Successfully analyzed {file_name} and stored in FAISS index", metrics
    except Exception as e:
        return None, f"Error processing file: {str(e)}", {}

# Create the Gradio interface
with gr.Blocks(css=custom_css, js=custom_js, theme=gr.themes.Soft()) as demo:
    # Header
    gr.HTML("""
    <header class="header">
        <div class="container">
            <div class="logo">
                <i class="fas fa-code logo-icon"></i>
                <span class="logo-text">Tech-Vision AI</span>
            </div>
        </div>
    </header>
    """)
    
    with gr.Row(elem_classes="container main-content"):
        # Sidebar
        with gr.Column(scale=1, min_width=320, elem_classes="sidebar"):
            with gr.Group(elem_classes="sidebar-section"):
                gr.Markdown("### Upload Code")
                file_input = gr.File(
                    label="Drag & drop your code file here",
                    file_types=[".py", ".js", ".java", ".cpp", ".c", ".cs", ".php", ".rb", ".go", ".ts"],
                    elem_classes="file-upload"
                )
                analyze_btn = gr.Button("Analyze Code", elem_classes="primary-button")
                
                model_dropdown = gr.Dropdown(
                    choices=["llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
                    value="llama3-70b-8192",
                    label="Select Model"
                )
            
            with gr.Group(elem_classes="sidebar-section"):
                gr.Markdown("### Developer Tools")
                with gr.Tabs():
                    with gr.TabItem("GitHub Search"):
                        repo_query = gr.Textbox(label="Search Query")
                        with gr.Row():
                            language = gr.Dropdown(
                                choices=["any", "JavaScript", "Python", "Java", "C++"],
                                value="any",
                                label="Language"
                            )
                            min_stars = gr.Dropdown(
                                choices=["0", "10", "50", "100", "1000"],
                                value="0",
                                label="Min Stars"
                            )
                        repo_search_btn = gr.Button("Search", elem_classes="primary-button")
                    
                    with gr.TabItem("Stack Overflow"):
                        stack_query = gr.Textbox(label="Search Query")
                        tag = gr.Dropdown(
                            choices=["any", "python", "javascript", "java"],
                            value="any",
                            label="Tag"
                        )
                        stack_search_btn = gr.Button("Search", elem_classes="primary-button")
        
        # Main Area
        with gr.Column(scale=3):
            with gr.Tabs(elem_classes="tabs"):
                with gr.TabItem("Code Analysis"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Code Metrics")
                            metrics = gr.JSON(label="")
                        with gr.Column():
                            gr.Markdown("### Recommendations")
                            recommendations = gr.Markdown()
                
                with gr.TabItem("GitHub Results"):
                    repo_results = gr.Markdown()
                
                with gr.TabItem("Stack Results"):
                    stack_results = gr.Markdown()
            
            # Chat Section
            with gr.Group(elem_classes="chat-section"):
                with gr.Row(elem_classes="chat-header"):
                    gr.Markdown("### Tech Assistant")
                    clear_btn = gr.Button("Clear", elem_classes="chat-control-btn")
                
                chatbot = gr.Chatbot(
                    height=400,
                    elem_classes="chat-messages",
                    show_copy_button=True,
                    type="messages"
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask about your code...",
                        show_label=False,
                        elem_classes="chat-input"
                    )
                    send_btn = gr.Button("Send", elem_classes="primary-button")

    # Add event handlers (implementation details in previous messages)
    # ... (event handlers remain the same)

if __name__ == "__main__":
    demo.launch() 