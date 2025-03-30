from flask import Flask, render_template, request, jsonify
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_groq import ChatGroq
import os


app = Flask(__name__)


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY")


wiki_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=500)
arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=500)

wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)


llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


prompt = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template=(
        "You are a highly intelligent and clear AI assistant. Answer with beautifully formatted, structured bullet points, using concise and meaningful explanations."
        " Add spacing, line breaks, and clean formatting for readability. Use \"bold keywords\" for emphasis."
        "\n\nChat History:\n{chat_history}\n\nUser: {user_input}\nAI:"
    )
)


chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message")
    
    # Generate structured response with beautiful formatting for UI
    response = chain.run(user_input)
    formatted_response = f"<div class='chat-response'>\n\n" + '\n'.join([f"<div><strong>{line.strip()}</strong></div>" if ':' in line else f"<div>{line.strip()}</div>" for line in response.split('\n')]) + "\n</div>"
    
    return jsonify({"response": formatted_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
