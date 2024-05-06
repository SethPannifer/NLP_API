from fastapi import FastAPI, Request, Form, Response
import subprocess
import numpy as np
import json
import pickle
import torch
import spacy
import tabulate
import time
import pandas as pd
import csv
from collections import Counter
from transformers import AutoModelForTokenClassification, AutoTokenizer

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from datetime import datetime

app = FastAPI()
#----------------------------------------------------------------------------------------------------------------------------------------------------------

class NERtask:
    def __init__(self, model):
        start_time = time.time()
        #Load Logistic Regression model
        print('Loading Logistic Regression Model')
        with open(model, 'rb') as f:
            self.LRModel = pickle.load(f)

        #Load RoBERTa model
        print('Loading Vectorization Model')
        model_checkpoint = "surrey-nlp/roberta-large-finetuned-abbr"
        self.RobBERTa_model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=5)
        self.RobBERTa_model.classifier = torch.nn.Identity()
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        print(f'Took {time.time() - start_time:.2f} seconds to load Models')
    
    def Tokenize(self, input):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(input)
        return [token.text for token in doc]
    
    def Vectorize(self, tokens):
        tokenized_input = self.tokenizer(tokens, is_split_into_words=True, truncation=True)
        vec_tokens = self.RobBERTa_model(torch.tensor(tokenized_input['input_ids']).unsqueeze(0)).logits.detach().squeeze().numpy()
        token_positions = tokenized_input.word_ids(batch_index=0)
        return vec_tokens, token_positions

    #Predict Written in Class Function so that all model initializations are performed once.
    def predict(self, input):
        start_time = time.time()
        tokens = self.Tokenize(input)
        vec_tokens, token_positions = self.Vectorize(tokens)
        results = self.LRModel.predict(vec_tokens[1:-1]) #We remove the CLS and SEP tokens from the predictions

        #Map Predictions to their respective positions
        position_label_map = dict()
        for pos, token in zip(token_positions[1:-1], results):
            if pos not in position_label_map:
                position_label_map[pos] = [token]
            else:
                position_label_map[pos].append(token)

        #Tokens may have different predictions due to subword tokenization. This code block merges them.
        for pos in position_label_map:
            if len(position_label_map[pos]) > 1:
                most_common,_ = Counter(position_label_map[pos]).most_common(1)[0]
                position_label_map[pos] = most_common
        
        token_label_map = {pos:(tok,lab) for pos,(tok,lab) in enumerate(zip(tokens, position_label_map.values()))}
        Results_Dataframe = pd.DataFrame(token_label_map)
        total_time = time.time() - start_time
        print(f'\nUser Input: {input}')
        print('\nNER Predictions:')
        print(f'\nTotal Time Taken: {total_time:.2f} seconds')
        print(tabulate.tabulate((Results_Dataframe), headers='keys', tablefmt='pretty', showindex='never'))
        return Results_Dataframe, total_time

#----------------------------------------------------------------------------------------------------------------------------------------------------------
LR_model_file = 'LR_NERtask_model.sav'
ner = NERtask(LR_model_file)
templates = Jinja2Templates(directory="templates")


@app.post("/run", response_class=HTMLResponse)
async def read_item(request:Request, input_text: str = Form(...)):

    LR_model_file = 'LR_NERtask_model.sav'
    Input_Text_1 = input_text 
    ner = NERtask(LR_model_file)
    data_test,time_taken = ner.predict(Input_Text_1)


    
    current_date = datetime.now()

    to_write = [data_test.iloc[0].tolist(), data_test.iloc[1].tolist(), time_taken, current_date]
    print(to_write)
    #create new csv if csv does not exist
    output_log_path = "Log_file.csv"
    with open(output_log_path, "a", newline='') as output_log:
        csv_write = csv.writer(output_log)
        csv_write.writerow(to_write) 

    data = {"title": "Run endpoint", "content1": to_write[0], "content2": to_write[1], "content3": to_write[2], "content4": to_write[3]}
    

    return Response(content=json.dumps(str(to_write)), media_type="application/json")

@app.get("/train")
async def run_train_bash():
    try:
        result = subprocess.check_output(["bash", "./train.sh"], stderr=subprocess.STDOUT)
        return {"output": result.decode("utf-8")}
    except subprocess.CalledProcessError as e:
        return {"error": e.output.decode("utf-8")}

@app.get("/CICD")
async def run_CICD_bash():
    try:
        result = subprocess.check_output(["bash", "./train.sh"], stderr=subprocess.STDOUT)
        return {"output": result.decode("utf-8")}
    except subprocess.CalledProcessError as e:
        return {"error": e.output.decode("utf-8")}
    
@app.get("/Hello")
async def hello():
    return "congrats on finding our hello world: Error 404"