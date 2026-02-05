# docbot
A vertical chatbot for querying documents using LLM and LlamaIndex


## Setup
Make sure ollama was installed on local machine and required model file was downloaded.
1. Clone the project to local: ```git clone git@github.com:everhettoo/docbot.git```
2. Create a virtual environment (if needed): ```python -m venv venv``` 
3. Activate the virtual environment: ```source venv/bin/activate```
4. Install required packages: ```pip install -r requirements.txt```
5. To start ollama with the selected model: ```ollama run llama3.1:8b-instruct```
6. To start the streamlit app: ```streamlit run app.py```
