from tabulate import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_metric
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import time
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
import pickle
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
import transformers
import torch
import os

#Device selection
if torch.cuda.is_available():
        device = torch.device("cuda")
elif torch.backends.mps.is_available():
        device = torch.device("mps")
else:
        device = torch.device("cpu")
print("Computing device:",device)

def _parser():
    parser = argparse.ArgumentParser(description='Train Model for Named Entity Recognition Task')
    parser.add_argument('--d', type=str, help='Dataset Name', default="surrey-nlp/PLOD-CW")
    parser.add_argument('--s', type=int, help='Dataset Subset Size', default=0)
    parser.add_argument('--n', type=str, help='Input Name', default='NERtask')
    return parser.parse_args()


class TrainModel:
    def __init__(self,subset):
        self.data_subset = subset
        self.result_dir = 'Training_Results'
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
    #Load Sequence Evaluation Metric
    def calculate_results(self, y_true, y_pred):
        metric = load_metric("seqeval", trust_remote_code=True)
        print(f"\nClassification Report\n")
        #Print Results using Classification Report Function
        print(classification_report(y_true, y_pred))

        #Compute Results using Sequence Evaluation Metric
        metric_results = metric.compute(predictions = [y_pred], references = [y_true])
        df = pd.DataFrame([metric_results['AC'], metric_results['LF'], metric_results['O']],index=['AC','LF','O'])
        print(tabulate(df, headers='keys', tablefmt='psql'))
        #Overall Results
        df = pd.DataFrame([metric_results['overall_precision'], 
                        metric_results['overall_recall'], 
                        metric_results['overall_f1'], 
                        metric_results['overall_accuracy']],
                        index=['Overall Precision','Overall Recall','Overall F1','Overall Accuracy'])
        #save results
        with open(f'{self.result_dir}/classification_report_{time.strftime("%Y%m%d-%H%M")}.txt', 'w') as f:
            f.write(f"\nClassification Report\n")
            f.write(classification_report(y_true, y_pred))
            f.write(tabulate(df, headers='keys', tablefmt='psql'))
        return metric_results

    #Confusion Matrix Function obtained from https://github.com/surrey-nlp/PLOD-AbbreviationDetection/blob/main/nbs/fine_tuning_abbr_det.ipynb
    def plot_cm(self, y_true, y_pred, name, figsize=(5,5)):
        #Get Confusion Matrix Values
        cm = confusion_matrix(y_true, y_pred)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                #Annotated Confusion Matrix with Percentage
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
        cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        fig, ax = plt.subplots(figsize=figsize)
        plt.title(f'Confusion Matrix for {name}')
        #Use seaborn heatmap to plot the confusion matrix
        sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax).figure
        plt.savefig(f'{self.result_dir}/{name}_CM_{time.strftime("%Y%m%d-%H%M")}.png')

    def plot_roc(self, y_true, y_prob, name):
        #Use Label Binarizer to binarize the labels so it can be used in the roc_curve function
        lb = LabelBinarizer().fit(y_true)
        y_true = lb.transform(y_true)
        plt.figure(figsize=(8, 6))
        for i in range(len(lb.classes_)):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{lb.classes_[i]} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Chance Level (AUC = 0.50)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve with AUC for {name}')
        plt.legend(loc='lower right')
        plt.savefig(f'{self.result_dir}/{name}_ROC_{time.strftime("%Y%m%d-%H%M")}.png')

    #Tokenization code obtained from https://github.com/surrey-nlp/NLP-2024/blob/main/lab06/BERT-finetuning.ipynb 
    def tokenize_and_align_labels(self,short_dataset, list_name):
        tokenizer = AutoTokenizer.from_pretrained("surrey-nlp/roberta-large-finetuned-abbr", add_prefix_space=True)
        assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
        tokenized_inputs = tokenizer(short_dataset["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(list_name):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def remove_unknown_tokens(self, tokens, labels):
        new_tokens, new_labels = [], []
        for token,label in zip(tokens,labels):
            if label != 'Unknown':
                new_tokens.append(token)
                new_labels.append(label)
        return new_tokens, new_labels

    def vectorize(self,dataset):
        start_time = time.time()
        print('Loading Dataset')
        if self.data_subset == 0:
            train_dataset = load_dataset(dataset, trust_remote_code=True,split="train")
            test_dataset = load_dataset(dataset, trust_remote_code=True,split="test")
        else:
            train_dataset = load_dataset(dataset, trust_remote_code=True,split=f"train[:{self.data_subset}]")
            test_dataset = load_dataset(dataset, trust_remote_code=True,split=f"test[:{self.data_subset}]")
        label_encoding = {"B-O": 0, "B-AC": 1, "B-LF": 2, "I-LF": 3}
        label_list = []
        for sample in train_dataset["ner_tags"]:
            label_list.append([label_encoding[tag] for tag in sample])

        test_label_list = []
        for sample in test_dataset["ner_tags"]:
            test_label_list.append([label_encoding[tag] for tag in sample])

        print('Performing Tokenization...')
        tokenized_datasets = self.tokenize_and_align_labels(train_dataset, label_list)
        tokenized_test_datasets = self.tokenize_and_align_labels(test_dataset, test_label_list)

        #Initialize the model with the pretrained model check point
        RobBERTa_model = AutoModelForTokenClassification.from_pretrained("surrey-nlp/roberta-large-finetuned-abbr", num_labels=5, device_map=device)
        #Set the classifier to Identity to get the word embeddings
        RobBERTa_model.classifier = torch.nn.Identity()
        print('Performing Vectorization...')
        train_vec_tokens = [RobBERTa_model(torch.tensor(row).unsqueeze(0).to(device)).logits.detach().squeeze().cpu().numpy() for row in tokenized_datasets['input_ids']]
        test_vec_tokens = [RobBERTa_model(torch.tensor(row).unsqueeze(0).to(device)).logits.detach().squeeze().cpu().numpy() for row in tokenized_test_datasets['input_ids']]
        reverse_label_encoding = {0: "B-O", 1: "B-AC", 2: "B-LF", 3: "I-LF", -100: "Unknown"}
        flatten_train_vec_tokens = [item for sublist in train_vec_tokens for item in sublist]
        flatten_test_vec_tokens = [item for sublist in test_vec_tokens for item in sublist]
        flatten_train_labels = [reverse_label_encoding[item] for sublist in tokenized_datasets['labels'] for item in sublist]
        flatten_test_labels = [reverse_label_encoding[item] for sublist in tokenized_test_datasets['labels'] for item in sublist]
        flatten_train_vec_tokens, flatten_train_labels = self.remove_unknown_tokens(flatten_train_vec_tokens, flatten_train_labels)
        flatten_test_vec_tokens, flatten_test_labels = self.remove_unknown_tokens(flatten_test_vec_tokens, flatten_test_labels)
        print('Vectorization Complete. Initializing Retraining...')
        print(f"Vectorization Time: {time.time()-start_time:.2f} seconds")
        return flatten_train_vec_tokens, flatten_test_vec_tokens, flatten_train_labels, flatten_test_labels


    def train(self, train_vec_tokens, train_labels):
        start_time = time.time()
        #Initialize a basic multiclass logistic regression model
        #Use the 'saga' solver for faster convergence
        #Use the 'multinomial' option for multiclass classification
        #Use the 'random_state' parameter for reproducibility
        model = LogisticRegression(multi_class='multinomial', solver='saga', random_state=1)
        model.fit(train_vec_tokens, train_labels)
        print(f"Training Time: {time.time()-start_time:.2f} seconds")
        return model

    def train_and_eval(self, train_vec_tokens, test_vec_tokens, train_labels, test_labels, input_name):
        model = self.train(train_vec_tokens, train_labels)
        predictions = model.predict(test_vec_tokens)
        pred_prob = model.predict_proba(test_vec_tokens)
        results = self.calculate_results(test_labels, predictions)
        self.plot_cm(test_labels, predictions, name=input_name)
        self.plot_roc(test_labels, pred_prob, input_name)
        filename = f'LR_{input_name}_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        return results, filename

if __name__ == '__main__':
    #Arguments can be passed into the code
    args = _parser()
    train_model = TrainModel(args.s)
    train_vec_tokens,test_vec_tokens, train_labels, test_labels = train_model.vectorize(args.d)
    results, model_path = train_model.train_and_eval(train_vec_tokens, test_vec_tokens, train_labels, test_labels, args.n)