# EasyChatBI
EasyChatBI: A backend service for data analysis built on ​FastAPI​ and ​Qwen Max​ (Alibaba Cloud's large language model). It supports uploading user-provided CSV or Excel files, analyzes data through ​natural language queries, and generates results in visual charts or tabular formats

Core Functional Modules​

​I. Application Initialization & Configuration​
Building web services using the ​FastAPI framework​
Configuring ​CORS support​ for cross-domain access
Setting up dedicated ​file upload directories​
Enabling ​Matplotlib Chinese font support​ with cross-platform compatibility (Windows/Linux/macOS)
Initializing ​Qwen Max API keys​ and model parameters

​II. Data Structure Definition​
​**AnalysisState**: A TypedDict defining the LangGraph state machine’s structure, including:
Session ID, user queries, file paths
DataFrame objects, metadata
Generated code snippets, execution results

​III. Core Processing Flow (LangGraph Workflow)​​
The LangGraph workflow comprises four key nodes:
​**load_dataframe**:
Loading CSV/Excel files into ​pandas DataFrames​
Supporting multiple encodings (utf-8, gb18030, gbk)
Auto-generating table metadata (column names, row counts)

​**parse_query**:
Using ​Qwen Max​ to parse natural language queries
Constructing prompt templates with table metadata + user questions
Generating ​Python code​ + ​visualization types​ (handling JSON/code-block responses)

​**execute_code**:
Safely executing Python code in ​restricted environments​
Processing visualizations (saving as image files)
Handling tabular/text results

​**generate_suggestions**:
Proposing ​3 follow-up questions​ based on query results

​IV. API Endpoints​
​File Upload (/upload)​​
Accepting CSV/Excel uploads
Generating unique ​session IDs​
Storing file paths globally

​Data Analysis (/analyze)​​
Executing full LangGraph workflows via session ID + NLP queries
Returning analysis results, suggested questions, and generated code

​Frontend Routing (/)​​
Serving static HTML pages

​V. Key Features​
​Multi-format support: CSV/Excel file ingestion
​Robust encoding handling: Auto-detection of CSV encodings
​Automated visualization: Matplotlib/Seaborn chart generation
​Context-aware suggestions: AI-recommended follow-up queries
​Secure execution: Sandboxed code environments

​VI. Technical Stack​
​Backend: FastAPI
​AI Engine: Qwen Max (Alibaba Cloud)
​Data Processing: pandas
​Visualization: Matplotlib, Seaborn
​Workflow Engine: LangGraph
​Frontend: Static HTML
