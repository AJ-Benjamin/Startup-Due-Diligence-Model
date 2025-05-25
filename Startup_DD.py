"""The first frontier model I designed is a helpful assistant powered by the OpenAI  gpt-4o-mini. With interactive textbox to receive user input message and a clear and submit button to erase or let her get to work. Thanks @gradio. I tested by asking "How can I be rich?" and she delivered. 

I redesigned the frontier model but powered by Anthropic claude-3-haiku-2022… model.

But, what about a model but you get to choose who runs it, OpenAI or Claude? And I created it

Finally, to create a likely to-be-used model, I created a model that
analyzes the contents of a company website landing page and creates a short brochure about the company for prospective customers, investors and recruits. 

This was achieved through web scrapping by using @beautiful soup and slapping in our frontier model with Gradio UI which have a GPT/Claude download to select who analyze the content of the website.
"""

# imports

import os
import requests
from bs4 import BeautifulSoup
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai
import anthropic

import gradio as gr

# Load environment variables in a file callled .env

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Connect to OpenAI, Anthropic and Google

openai = OpenAI()

claude = anthropic.Anthropic()

google.generativeai.configure()

system_message = "You are an assistant that analyzes the contents of a company website landing page and creates a short brochure about the company for prospective customers, investors and recruit do their due diligence on any company to know if it is a good fit for them to work with or invest"

# Runs only on GPT model

def stream_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    stream = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result
		
view = gr.Interface(
    fn=stream_gpt,
    inputs=[gr.Textbox(label="Your message:")],
    outputs=[gr.Markdown(label="Response")],
    flagging_mode = "never"
)
view.launch()




# Runs on only Claude

def stream_claude(prompt):
    result = claude.messages.stream( 
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.7,
        system=system_message,
        messages=[
            {"role": "user","content": prompt},
        ],
    )
    response = ""
    with result as stream:
        for text in stream.text_stream:
            response += text or ""
            yield response
			
view = gr.Interface(
    fn=stream_claude,
    inputs=[gr.Textbox(label="Your message:")],
    outputs=[gr.Markdown(label="Response:")],
    flagging_mode="never"
)
view.launch()



# A model option of choosing between gpt and claude
def stream_model(prompt, model):
    if model=="GPT":
        result = stream_gpt(prompt)
    elif model=="Claude":
        result = stream_claude(prompt)
    else:
        raise ValueError("Unknown model")
    # for chunk in result:
    #     yield chunk
    yield from result
        
view = gr.Interface(
    fn=stream_model,
    inputs=[gr.Textbox(label="Your message:"), gr.Dropdown(["GPT", "Claude"], label="Select model")],
    outputs=[gr.Markdown(label="Response:")],
    flagging_mode="never"
)
view.launch()



### Building a company's brochure generator

# A class to represent a webpage]
class Website:
    url: str
    title: str
    text: str

    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.body = response.content
        soup = BeautifulSoup(self.body, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"
		
system_prompt = "You are an assistant that analyzes the contents of a company website landing page \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown."

def stream_brochure(company_name, url, model):
    prompt = f"Please generate a company brochure  for {company_name}. Here is their landing page:\n"
    prompt += Website(url).get_contents()
    if model=="GPT":
        result = stream_gpt(prompt)
    elif model=="Claude":
        result = stream_claude(prompt)
    else:
        raise ValueError("Unknown model")
    for chunk in result:
        yield chunk

view = gr.Interface(
    fn=stream_brochure,
    inputs=[
        gr.Textbox(label="Company name:"),
        gr.Textbox(label="Landing age URL:"),
        gr.Dropdown(["GPT", "Claude"], label="Select model")],
    outputs=[gr.Markdown(label="Brochure:")],
    flagging_mode="never"
)
view.launch(share=True, inbrowser=True)

