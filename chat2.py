from customtkinter import *
from datetime import datetime
import json
from pinecone import Pinecone
from transformers import BertTokenizer, BertModel
import torch
import google.generativeai as genai

## the following function will generate text embeddings using bert model
def question_embeddings(message, model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name) ### here the function loads the pre-trained bert model
    
    inputs = tokenizer(message, return_tensors='pt', truncation=True, padding=True) ## the input message is tokenized, preparing the input to be fed into the model
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    question_vectors = outputs.last_hidden_state[:, 0, :].squeeze().numpy() ## Extracts the embeddings (vectors) from the model's output.
    return question_vectors

def query_db(question_vectors, k, index_name): ## Queries the index for similar vectors, returning the top k results.
    pc = Pinecone(api_key='09c53ad3-28c3-4a7f-88fa-68260478262e')
    index = pc.Index(index_name)
    
    metadata_dict = index.query(
        vector=question_vectors.tolist(),
        top_k=k,
        include_metadata=True
    )
    ls = []
    for i in metadata_dict['matches']: ### Iterates through the matched results
        ls.append(i['metadata']['text'])
    full = ', '.join(ls)
    return full

def gemini_response(question, context): ## defines a function to generate response using Gemini
    genai.configure(api_key='AIzaSyDTCE7yEZcy5YED7kZiGbq9iW2Qimvhjqs')
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(
        f'You are a helpful and informative bot that answers questions using text from the reference passage included below. \
        Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
        However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
        strike a friendly and conversational tone. \
        If the passage is irrelevant to the answer, you may ignore it. \
        QUESTION: {question} \
        PASSAGE: {context}'
    )
    return response.text

# Initialize the main application window
app = CTk()
app.title('Financial Advisor Chatbot')
app.geometry("800x700")
app.resizable(False, False)

# Set the color theme to green
set_default_color_theme("green")

# Function to clear the chat messages
def clear_message():
    for widget in chat_frame.winfo_children(): ## Iterates through all child widgets in the chat frame
        widget.destroy() ## Destroys each widget (clearing messages)
    canvas.yview_moveto(0) ## Scrolls the canvas to the top

# Function to save chat history
def save_chat_history():
    history = []
    for widget in chat_frame.winfo_children():
        if isinstance(widget, CTkLabel):
            history.append(widget.cget("text")) ## Appends the text from each label to the history list
    with open("chat_history.json", "w") as f:
        json.dump(history, f)

# Function to load chat history
def load_chat_history():
    try:
        with open("chat_history.json", "r") as f:
            history = json.load(f)
            for text in history:
                label = CTkLabel(master=chat_frame, text=text, anchor='w', fg_color='#f0f0f0', wraplength=700, pady=10, padx=10)
                label.pack(anchor='w', pady=5, padx=5)
    except FileNotFoundError:
        pass

# Function to handle sending messages
def send_message(event=None, message=None):
    if message is None:
        message = user_entry.get().strip()
    
    if message == '':
        return
    
    timestamp = datetime.now().strftime('%H:%M')
    user_label = CTkLabel(master=chat_frame, text=f'{message}  {timestamp}', anchor='e', fg_color='#e1ffc7', wraplength=700, pady=10, padx=10)
    user_label.pack(anchor='e', pady=5, padx=5)

    bot_response = get_bot_response(message)
    display_bot_response(bot_response, timestamp)
    
    user_entry.delete(0, 'end')
    canvas.yview_moveto(1)

## Define the function to organize the chat reposones 

def get_bot_response(user_input):
    print(f"User Input: {user_input}")
    
    if 'hello' in user_input.lower():
        return 'Hello, how can I assist you today?'
    elif 'tell me more about this bot' in user_input.lower():
        return 'I\'m a financial chatbot advisor made to help you invest your money effectively.'
    elif user_input.lower() == 'clear':
        clear_message()
        return ''
    else:
        question_vectors = question_embeddings(user_input, "bert-base-uncased")
        print(f"Question Vectors: {question_vectors}")
        
        context = query_db(question_vectors, 3, 'bert-test9')
        print(f"Context: {context}")
        
        response = gemini_response(user_input, context)
        print(f"Bot Response: {response}")
        
        return response

def display_bot_response(response, timestamp, index=0):
    if index == 0:
        bot_label = CTkLabel(master=chat_frame, text='', anchor='w', fg_color='#f0f0f0', wraplength=700, pady=10, padx=10)
        bot_label.pack(anchor='w', pady=5, padx=5)
        app.bot_label = bot_label
        app.bot_text = ''
    if index < len(response):
        app.bot_text += response[index]
        app.bot_label.configure(text=f'{app.bot_text} {timestamp}')
        index += 1
        app.after(25, display_bot_response, response, timestamp, index)

# Header
header_text = CTkLabel(master=app, text='Financial Advisor Chatbot', font=('Impact', 24))
header_text.pack(pady=20)

# Chat background frame with canvas and scrollbar
chat_bg = CTkFrame(master=app, fg_color='#f0f0f0')
chat_bg.pack(fill='both', expand=True, padx=10, pady=10)

canvas = CTkCanvas(chat_bg)
canvas.pack(side='left', fill='both', expand=True)

chat_scroll = CTkScrollbar(chat_bg, command=canvas.yview)
chat_scroll.pack(side='right', fill='y')

chat_frame = CTkFrame(canvas, fg_color='#f0f0f0')
canvas.create_window((0, 0), window=chat_frame, anchor='nw', width=780)

chat_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

canvas.configure(yscrollcommand=chat_scroll.set)

# Load chat history
load_chat_history()

# Entry background frame
entry_bg = CTkFrame(master=app, fg_color='#f0f0f0')
entry_bg.pack(fill='x', padx=10, pady=10)

# User entry
user_entry = CTkEntry(master=entry_bg, placeholder_text="Enter message...", font=('Arial', 16), height=40, width=400)
user_entry.pack(side='left', fill='x', expand=True, padx=10, pady=10)
user_entry.bind("<Return>", send_message)

# Send button
send_button = CTkButton(master=entry_bg, text='Send', command=send_message, width=60, height=40, font=('Arial', 12))
send_button.pack(side='left', padx=10, pady=10)

# Clear button
clear_button = CTkButton(master=entry_bg, text='Clear', command=clear_message, width=60, height=40, font=('Arial', 12))
clear_button.pack(side='left', padx=10, pady=10)

# Save button
save_button = CTkButton(master=entry_bg, text='Save Chat', command=save_chat_history, width=80, height=40, font=('Arial', 12))
save_button.pack(side='left', padx=10, pady=10)

# Quick reply buttons
quick_replies = CTkFrame(master=app, fg_color='#f0f0f0')
quick_replies.pack(fill='x', padx=10, pady=10)

quick_reply2 = CTkButton(master=quick_replies, text='About me?', command=lambda: send_message(message='tell me more about this bot'), width=100, height=40, font=('Arial', 12))
quick_reply2.pack(side='left', padx=10, pady=10)

app.mainloop()
