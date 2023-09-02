from flask import Flask, render_template, jsonify, request
import openai
import langchain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = Flask(__name__)


@app.route('/')
def index():
  return render_template("index.html")

@app.route('/generate', methods=['GET', 'POST'])
def generate():
      if request.method == 'POST':
        llm = OpenAI(openai_api_key="api-key", temperature=0.3)
        prompt = request.json.get('prompt')
        
        prompt1 = PromptTemplate.from_template("Generate a blog on title {title}?")
        chain = LLMChain(llm = llm, prompt= prompt1)
        output = chain.run(prompt)
        print(output)
        
        return output

app.run(host='0.0.0.0', port=5000)
