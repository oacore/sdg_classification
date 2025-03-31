import os
import pandas as pd
import random
import pickle
import ast
import numpy as np
import torch

from utils.configs_util import load_config
import logging
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from transformers import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    precision_score, recall_score, f1_score, hamming_loss, jaccard_score,
    log_loss, average_precision_score
)
from es_core import query_es_by_title, query_es_by_id
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, evaluation
from torch.utils.data import DataLoader, Dataset
from data import DATA_DIR
from models import MODEL_DIR
from output import OUTPUT_DIR
from utils.constants import NON_OVERLAPPING_SDGS
from warnings import simplefilter
from sklearn.model_selection import train_test_split
from arg_parser import get_args
import numpy as np
from metadata_extraction import get_core_id, CORESingleMetaDataExtraction

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

class InputExampleDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class DescriptionDataLoader:
    def __init__(self, label_desc_dir, sbert_model):
        self.label_desc_dir = label_desc_dir
        self.sbert_model = sbert_model
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(ENGLISH_STOP_WORDS)
        self.sdg_definitions = self.get_sdg_definition()

    def read_descriptions(self, file_path):
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        return df['description'].tolist()

    def get_sdg_definition(self):
        sdg_definitions = {}
        for sdg_folder in os.listdir(self.label_desc_dir):
            sdg_path = os.path.join(self.label_desc_dir, sdg_folder)
            if os.path.isdir(sdg_path) and not sdg_folder.startswith('.'):
                sdg_definitions[sdg_folder] = {"Goal": [], "Targets": [], "Indicators": []}
                for category in sdg_definitions[sdg_folder].keys():
                    file_path = os.path.join(sdg_path, f'{category.lower()}.txt')
                    if os.path.exists(file_path) and file_path.endswith('.txt'):
                        descriptions = self.read_descriptions(file_path)
                        sdg_definitions[sdg_folder][category].extend(descriptions)
        return sdg_definitions


class MultiLabelDatasetLoader:
    def __init__(self, multi_label_data_dir):
        self.multi_label_data_dir = multi_label_data_dir
        #self.sbert_model = sbert_model


    def create_labels(self, row):
        labels = []
        for i in range(1, 18):
            col_name = f'SDG-{i:02d}'
            if row[col_name] == 1:
                labels.append(col_name.replace('-', ''))
        return labels

    def read_dataset(self):

        df = pd.read_csv(os.path.join(self.multi_label_data_dir, 'sdg_knowledge_hub.csv'), encoding='utf-8',
                         engine='python')

        df['labels'] = df.apply(self.create_labels, axis=1)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        return train_df, test_df


class MultiLabelSyntheticDatasetLoader:
    def __init__(self, multi_label_data_dir):
        self.multi_label_data_dir = multi_label_data_dir
        #self.sbert_model = sbert_model

    def safe_literal_eval(self, val):
        if val is None:
            return None  # or return an empty list [] if that suits your needs better
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError) as e:
            print(f"Skipping malformed value: {val}. Error: {e}")
            return None  # or handle it in a way that makes sense for your data

    def read_dataset(self):

        df = pd.read_csv(os.path.join(self.multi_label_data_dir, 'synthetic_final.tsv'), sep = '\t', encoding='utf-8',
                         engine='python')

        df['title'].fillna('', inplace=True)
        df['abstract'].fillna('', inplace=True)
        df['text'] = df['title'] + '. ' + df['abstract']
        df['labels'] = df['labels'].apply(self.safe_literal_eval)
        #df['text'] = df['text'].str.strip('. ')

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        return train_df, test_df


class MultiLabelDatasetOROLoader:
    def __init__(self, multi_label_data_oro_dir):
        self.multi_label_data_oro_dir = multi_label_data_oro_dir

    def safe_literal_eval(self, val):
        if val is None:
            return None  # or return an empty list [] if that suits your needs better
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError) as e:
            print(f"Skipping malformed value: {val}. Error: {e}")
            return None  # or handle it in a way that makes sense for your data

    def read_dataset(self):

        df = pd.read_csv(os.path.join(self.multi_label_data_oro_dir, 'oro_gold_dataset.txt'),
                                            sep='\t', encoding='utf-8', engine='python')
        df['abstract'].fillna('', inplace=True)
        df['title'].fillna('', inplace=True)
        df['text'] = df['title'] + '. ' + df['abstract']
        df['labels'] = df['labels'].apply(self.safe_literal_eval)
        return df

class ORODataLoader:
    def __init__(self, oro_data_dir):
        self.oro_data_dir = oro_data_dir

    def read_dataset(self):

        df = pd.read_csv(os.path.join(self.oro_data_dir, 'oro_title_abstracts.txt'),
                                            sep='\t', encoding='utf-8', engine='python', on_bad_lines='skip')

        df['title'].fillna('', inplace=True)
        df['abstract'].fillna('', inplace=True)
        df['text'] = df['title'] + '. ' + df['abstract']

        return df

class DatasetLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_dataset(self):

        df = pd.read_csv(self.file_path, sep='\t', encoding='utf-8', engine='python', on_bad_lines='skip')

        # df = pd.read_csv(StringIO(os.path.join(self.oro_data_dir, 'oro_title_abstracts.txt')), sep='\t', encoding='utf-8',
        #                  engine='python', quoting=csv.QUOTE_ALL, escapechar='\\', header=None)

        # Ensure required columns are present
        required_columns = {'id', 'title', 'abstract', 'date'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Input file is missing required columns: {missing}")

        df['title'].fillna('', inplace=True)
        df['abstract'].fillna('', inplace=True)
        df['text'] = df['title'] + '. ' + df['abstract']

        return df

class DescriptionFineTuning:
    def __init__(self, sdg_definitions, non_overlapping_sdgs, sbert_model):
        self.sdg_definitions = sdg_definitions
        self.non_overlapping_sdgs = non_overlapping_sdgs
        self.sbert_model = sbert_model

    def sentence_pairs_generation(self, sdg):
        pairs = []
        goal = self.sdg_definitions[sdg]['Goal'][0]
        positives = self.sdg_definitions[sdg]['Targets']
        # Generating positive pairs
        for pos in positives:
            pairs.append(InputExample(texts=[goal, pos], label=1.0))

        # Number of positive pairs
        num_positives = len(positives)
        # Generating negative pairs
        negative_samples = []
        non_overlapping = self.non_overlapping_sdgs.get(sdg, [])
        for neg_sdg in non_overlapping:
            if neg_sdg in self.sdg_definitions:
                negatives = self.sdg_definitions[neg_sdg]['Targets']
                for neg in negatives:
                    negative_samples.append(neg)

        sampled_negatives = random.sample(negative_samples, num_positives)

        for neg in sampled_negatives:
            pairs.append(InputExample(texts=[goal, neg], label=0.0))

        return pairs

    def prepare_data(self):
        train_examples = []
        for sdg in self.sdg_definitions.keys():
            sdg_pairs = self.sentence_pairs_generation(sdg)
            train_examples.extend(sdg_pairs)
        return train_examples

    def train_model(self, train_examples):
        random.shuffle(train_examples)
        # S-BERT adaptation
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(self.sbert_model)
        self.sbert_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=10, show_progress_bar=True)
        return self.sbert_model


class MultiLabelSBERTFineTuning:
    def __init__(self, train_df, sbert_model):
        self.train_df = train_df
        #self.test_df = test_df
        self.sbert_model = sbert_model
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
        #self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.text = self.train_df.columns.values[3]
        self.labels_col = self.train_df.columns.values[23]


    def get_unique_labels(self, y_train):
        unique_labels = set()
        for labels in y_train:
            unique_labels.update(labels)
        return list(unique_labels)

    def sentence_pairs_generation(self, sentences, labels, pairs):

        label_sets = [set(label) for label in labels]
        for idxA in range(len(sentences)):
            current_sentence = sentences[idxA]
            label = label_sets[idxA]
            pos_indices = [idx for idx, lbl_set in enumerate(label_sets) if lbl_set == label and idx != idxA]
            if not pos_indices:

                pos_indices = [idx for idx, lbl_set in enumerate(label_sets) if lbl_set & label and idx != idxA]
                pos_idx = np.random.choice(pos_indices)
                pos_sentence = sentences[pos_idx]
                #label_pos = label_sets[idxB]
                pairs.append(InputExample(texts=[current_sentence, pos_sentence], label=1.0))

                if not pos_indices:
                    print("No positive sentence found.\n")
                    continue

            else:
                pos_idx = np.random.choice(pos_indices)
                pos_sentence = sentences[pos_idx]
                pairs.append(InputExample(texts=[current_sentence, pos_sentence], label=1.0))

            neg_indices = [idx for idx, lbl_set in enumerate(label_sets) if lbl_set != label and idx != idxA]
            if not neg_indices:
                print("No negative sentence found.\n")
                continue

            else:
                neg_idx = np.random.choice(neg_indices)
                neg_sentence = sentences[neg_idx]
                pairs.append(InputExample(texts=[current_sentence, neg_sentence], label=0.0))

        return (pairs)

    def sample_train_data(self, args):

        train_positive = []
        for sdg in range(1, 18):
            sdg_str = f'SDG{str(sdg).zfill(2)}'
            positive_df = self.train_df[self.train_df['labels'].apply(lambda labels: sdg_str in labels and len(labels) <= 7)]
            #sdg_negative_df = self.train_df[(self.train_df[sdgs] == sdg) & (self.train_df[labels] == False)]

            if len(positive_df) >= args.num_training:
                train_positive.append(positive_df.sample(n=args.num_training, random_state=args.seed))

        train_df_sample = pd.concat(train_positive).reset_index(drop=True)
        # train_df_negative_sample = pd.concat(balanced_train_negative).reset_index(drop=True)
        x_train = train_df_sample[self.text].values.tolist()
        y_train = train_df_sample[self.labels_col].values.tolist()

        return x_train, y_train

    def prepare_data(self):

        train_examples = []
        args = get_args()
        x_train, y_train = self.sample_train_data(args)
        for x in range(args.num_iter):
            train_examples = self.sentence_pairs_generation(x_train, y_train, train_examples)

        return x_train, y_train, train_examples

    def head_tail_truncation(self, text, max_length=512, head_tokens=128, tail_tokens=382):
        """
        Truncates the input text, keeping the first `head_tokens` and the last `tail_tokens` tokens,
        ensuring the total length does not exceed `max_length`.
        """
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > max_length:
            if head_tokens + tail_tokens > max_length:
                raise ValueError("Sum of head_tokens and tail_tokens exceeds max_length.")
            truncated_tokens = tokens[:head_tokens] + tokens[-tail_tokens:]
        else:
            truncated_tokens = tokens

        return self.tokenizer.convert_tokens_to_string(truncated_tokens)

    def sbert_finetuning(self, train_examples):
        #print(len(train_examples))
        random.shuffle(train_examples)
        truncated_train_examples = []
        for example in train_examples:
            truncated_texts = [
                self.head_tail_truncation(text) for text in example.texts
            ]
            truncated_train_examples.append(InputExample(texts=truncated_texts, label=example.label))

        # S-BERT adaptation
        #train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_dataloader = DataLoader(truncated_train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(self.sbert_model)
        self.sbert_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=10,
                                     show_progress_bar=True)

        return self.sbert_model


class MultiLabelSyntheticSBERTFineTuning:
    def __init__(self, train_df, sbert_model):
        self.train_df = train_df
        #self.test_df = test_df
        self.sbert_model = sbert_model
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
        #self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.text = self.train_df.columns.values[3]
        self.labels_col = self.train_df.columns.values[2]

    def get_unique_labels(self, y_train):
        unique_labels = set()
        for labels in y_train:
            unique_labels.update(labels)
        return list(unique_labels)

    def sentence_pairs_generation(self, sentences, labels, pairs):

        label_sets = [set(label) for label in labels]
        for idxA in range(len(sentences)):
            current_sentence = sentences[idxA]
            label = label_sets[idxA]
            pos_indices = [idx for idx, lbl_set in enumerate(label_sets) if lbl_set == label and idx != idxA]
            if not pos_indices:

                pos_indices = [idx for idx, lbl_set in enumerate(label_sets) if lbl_set & label and idx != idxA]
                pos_idx = np.random.choice(pos_indices)
                pos_sentence = sentences[pos_idx]
                #label_pos = label_sets[idxB]
                pairs.append(InputExample(texts=[current_sentence, pos_sentence], label=1.0))

                if not pos_indices:
                    print("No positive sentence found.\n")
                    continue

            else:
                pos_idx = np.random.choice(pos_indices)
                pos_sentence = sentences[pos_idx]
                pairs.append(InputExample(texts=[current_sentence, pos_sentence], label=1.0))

            neg_indices = [idx for idx, lbl_set in enumerate(label_sets) if lbl_set != label and idx != idxA]
            if not neg_indices:
                print("No negative sentence found.\n")
                continue

            else:
                neg_idx = np.random.choice(neg_indices)
                neg_sentence = sentences[neg_idx]
                pairs.append(InputExample(texts=[current_sentence, neg_sentence], label=0.0))

        return (pairs)

    def sample_train_data(self, args):

        train_positive = []
        for sdg in range(1, 18):
            sdg_str = f'SDG{str(sdg).zfill(2)}'
            positive_df = self.train_df[self.train_df['labels'].apply(lambda labels: sdg_str in labels)]
            #sdg_negative_df = self.train_df[(self.train_df[sdgs] == sdg) & (self.train_df[labels] == False)]

            if len(positive_df) >= args.num_training:
                train_positive.append(positive_df.sample(n=args.num_training, random_state=args.seed))

        train_df_sample = pd.concat(train_positive).reset_index(drop=True)
        # train_df_negative_sample = pd.concat(balanced_train_negative).reset_index(drop=True)
        x_train = train_df_sample[self.text].values.tolist()
        y_train = train_df_sample[self.labels_col].values.tolist()

        return x_train, y_train

    def prepare_data(self):

        train_examples = []
        args = get_args()
        x_train, y_train = self.sample_train_data(args)
        for x in range(args.num_iter):
            train_examples = self.sentence_pairs_generation(x_train, y_train, train_examples)

        return x_train, y_train, train_examples

    def head_tail_truncation(self, text, max_length=512, head_tokens=128, tail_tokens=382):
        """
        Truncates the input text, keeping the first `head_tokens` and the last `tail_tokens` tokens,
        ensuring the total length does not exceed `max_length`.
        """
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > max_length:
            if head_tokens + tail_tokens > max_length:
                raise ValueError("Sum of head_tokens and tail_tokens exceeds max_length.")
            truncated_tokens = tokens[:head_tokens] + tokens[-tail_tokens:]
        else:
            truncated_tokens = tokens

        return self.tokenizer.convert_tokens_to_string(truncated_tokens)

    def sbert_finetuning(self, train_examples):
        random.shuffle(train_examples)
        truncated_train_examples = []
        for example in train_examples:
            truncated_texts = [
                self.head_tail_truncation(text) for text in example.texts
            ]
            truncated_train_examples.append(InputExample(texts=truncated_texts, label=example.label))
        # S-BERT adaptation
        #train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

        train_dataloader = DataLoader(truncated_train_examples, batch_size=16, shuffle=True)
        train_loss = losses.CosineSimilarityLoss(self.sbert_model)
        self.sbert_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=10,
                                     show_progress_bar=True)

        return self.sbert_model

class MultiLabelOROSBERTFineTuning:
    def __init__(self, train_df, sbert_model):
        self.train_df = train_df
        self.sbert_model = sbert_model
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
        self.text = self.train_df.columns.values[4]
        self.labels_col = self.train_df.columns.values[3]

    def get_unique_labels(self, y_train):
        unique_labels = set()
        for labels in y_train:
            unique_labels.update(labels)
        return list(unique_labels)

    def sentence_pairs_generation(self, sentences, labels, pairs):

        label_sets = [set(label) for label in labels]
        for idxA in range(len(sentences)):
            current_sentence = sentences[idxA]
            label = label_sets[idxA]
            pos_indices = [idx for idx, lbl_set in enumerate(label_sets) if lbl_set == label and idx != idxA]
            if not pos_indices:

                pos_indices = [idx for idx, lbl_set in enumerate(label_sets) if lbl_set & label and idx != idxA]
                pos_idx = np.random.choice(pos_indices)
                pos_sentence = sentences[pos_idx]
                #label_pos = label_sets[idxB]
                pairs.append(InputExample(texts=[current_sentence, pos_sentence], label=1.0))

                if not pos_indices:
                    print("No positive sentence found.\n")
                    continue

            else:
                pos_idx = np.random.choice(pos_indices)
                pos_sentence = sentences[pos_idx]
                pairs.append(InputExample(texts=[current_sentence, pos_sentence], label=1.0))

            neg_indices = [idx for idx, lbl_set in enumerate(label_sets) if lbl_set != label and idx != idxA]
            if not neg_indices:
                print("No negative sentence found.\n")
                continue

            else:
                neg_idx = np.random.choice(neg_indices)
                neg_sentence = sentences[neg_idx]
                pairs.append(InputExample(texts=[current_sentence, neg_sentence], label=0.0))

        return (pairs)

    def sample_train_data(self, args):

        train_positive = []
        for sdg in range(1, 18):
            sdg_str = f'SDG{str(sdg).zfill(2)}'
            positive_df = self.train_df[
                self.train_df['human_annotator'].apply(lambda labels: sdg_str in labels)]
            #sdg_negative_df = self.train_df[(self.train_df[sdgs] == sdg) & (self.train_df[labels] == False)]

            if len(positive_df) >= args.num_training:
                train_positive.append(positive_df.sample(n=args.num_training, random_state=args.seed))

        train_df_sample = pd.concat(train_positive).reset_index(drop=True)
        #train_df_sample['combined_text'] = train_df_sample['title'] + '. ' + train_df_sample['abstract']
        #print(train_df_sample['combined_text'])
        # train_df_negative_sample = pd.concat(balanced_train_negative).reset_index(drop=True)
        x_train = train_df_sample[self.text].values.tolist()
        y_train = train_df_sample[self.labels_col].values.tolist()
        # print(len(x_train))
        # print(len(y_train))

        return x_train, y_train

    def prepare_data(self):

        train_examples = []
        args = get_args()
        x_train, y_train = self.sample_train_data(args)
        for x in range(args.num_iter):
            train_examples = self.sentence_pairs_generation(x_train, y_train, train_examples)

        return x_train, y_train, train_examples

    def head_tail_truncation(self, text, max_length=512, head_tokens=128, tail_tokens=382):
        """
        Truncates the input text, keeping the first `head_tokens` and the last `tail_tokens` tokens,
        ensuring the total length does not exceed `max_length`.
        """
        #print(text)
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > max_length:
            if head_tokens + tail_tokens > max_length:
                raise ValueError("Sum of head_tokens and tail_tokens exceeds max_length.")
            truncated_tokens = tokens[:head_tokens] + tokens[-tail_tokens:]
        else:
            truncated_tokens = tokens

        return self.tokenizer.convert_tokens_to_string(truncated_tokens)

    def sbert_finetuning(self, train_examples):

        random.shuffle(train_examples)
        truncated_train_examples = []
        for example in train_examples:

            truncated_texts = [
                self.head_tail_truncation(text) for text in example.texts
            ]
            truncated_train_examples.append(InputExample(texts=truncated_texts, label=example.label))

        # S-BERT adaptation
        #train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_dataloader = DataLoader(truncated_train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(self.sbert_model)
        self.sbert_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=10,
                                     show_progress_bar=True)

        return self.sbert_model

class LinearClassifierKnowledgeHub:
    def __init__(self, test_df, mlb=None):
        """
        Initializes the classifier, tokenizer, and dataset parameters.
        """
        self.linear_classifier = OneVsRestClassifier(LogisticRegression())
        self.mlb = mlb if mlb is not None else MultiLabelBinarizer()  # Allow external mlb for consistency
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
        self.test_df = test_df
        self.text = self.test_df.columns.values[3]   # Ensure column consistency
        self.labels_col = self.test_df.columns.values[23]  # Index for label column

    def head_tail_truncation(self, text, max_length=512, head_tokens=128, tail_tokens=382):
        """
        Truncates text to preserve head and tail tokens while maintaining max token length.
        """
        if not isinstance(text, str) or pd.isna(text):
            text = ""
        tokens = self.tokenizer.tokenize(text)
        truncated_tokens = tokens[:head_tokens] + tokens[-tail_tokens:] if len(tokens) > max_length else tokens
        return self.tokenizer.convert_tokens_to_string(truncated_tokens)

    def train_model(self, x_train, y_train, model):
        """
        Encodes training data and fits the classifier.
        """

        # self.mlb.fit(y_train)  # Fit only on y_train
        # y_train_binary = self.mlb.transform(y_train)
        y_train_binary = self.mlb.fit_transform(y_train)
        X_train = np.array(model.encode(x_train, convert_to_numpy=True))

        self.linear_classifier.fit(X_train, y_train_binary)
        return self.linear_classifier, self.mlb

    def eval_model(self, model, classifier):
        """
        Encodes evaluation data, performs predictions, and computes evaluation metrics.
        """
        threshold = 0.6
        y_eval = self.test_df[self.labels_col].tolist()
        y_eval_binary = self.mlb.transform(y_eval)

        # Apply truncation and encode using the model
        self.test_df['truncated_text'] = self.test_df[self.text].apply(lambda x: self.head_tail_truncation(x))
        x_eval = self.test_df['truncated_text'].tolist()
        X_eval = np.array(model.encode(x_eval, convert_to_numpy=True))

        if X_eval.shape[0] != y_eval_binary.shape[0]:
            raise ValueError(f"Mismatch: X_eval has {X_eval.shape[0]} samples, y_eval has {y_eval_binary.shape[0]} samples.")

        # Predict probabilities and apply threshold
        y_pred_proba = classifier.predict_proba(X_eval)
        y_pred = (y_pred_proba >= threshold).astype(int)

        return self.compute_metrics(y_eval_binary, y_pred, y_pred_proba, y_eval)

    def compute_metrics(self, y_eval_binary, y_pred, y_pred_proba, y_eval):
        """
        Computes classification metrics including accuracy, precision, recall, and log loss.
        """
        strict_accuracy = np.mean(np.all(y_pred == y_eval_binary, axis=1))
        weak_accuracy = np.mean(np.any(np.logical_and(y_pred, y_eval_binary), axis=1))

        num_classes = y_eval_binary.shape[1]
        APs = [average_precision_score(y_eval_binary[:, j], y_pred_proba[:, j]) for j in range(num_classes)]
        macro_mAP = np.mean(APs)
        weighted_mAP = np.average(APs, weights=np.sum(y_eval_binary, axis=0))
        micro_mAP = average_precision_score(y_eval_binary.ravel(), y_pred_proba.ravel())
        hamming = hamming_loss(y_eval_binary, y_pred)

        precision_micro = precision_score(y_eval_binary, y_pred, average='micro', zero_division=0)
        recall_micro = recall_score(y_eval_binary, y_pred, average='micro', zero_division=0)
        f1_micro = f1_score(y_eval_binary, y_pred, average='micro', zero_division=0)
        precision_macro = precision_score(y_eval_binary, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_eval_binary, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_eval_binary, y_pred, average='macro', zero_division=0)
        jaccard_sim = jaccard_score(y_eval_binary, y_pred, average='micro', zero_division=0)
        logloss = log_loss(y_eval_binary, y_pred_proba)

        # Extract only the predicted labels
        predictions = [[label for label, proba in zip(self.mlb.classes_, instance_probas) if proba >= 0.5]
                       for instance_probas in y_pred_proba]

        predictions_df = pd.DataFrame({
            "y_pred": predictions,  # List of predicted labels
            "y_actual": y_eval      # List of actual labels
        })

        return (strict_accuracy, weak_accuracy, hamming, precision_micro, recall_micro, f1_micro, precision_macro,
                recall_macro, f1_macro, jaccard_sim, logloss, macro_mAP, weighted_mAP, micro_mAP, APs), predictions_df

# class LinearClassifierKnowledgeHub:
#
#     def __init__(self, test_df):
#         self.linear_classifier = OneVsRestClassifier(LogisticRegression())
#         self.mlb = MultiLabelBinarizer()
#         self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
#         self.test_df = test_df
#         self.text = self.test_df.columns.values[3]
#         self.labels_col = self.test_df.columns.values[23]
#
#     def head_tail_truncation(self, text, max_length=512, head_tokens=128, tail_tokens=382):
#         """
#         Truncates the input text, keeping the first `head_tokens` and the last `tail_tokens` tokens,
#         ensuring the total length does not exceed `max_length`.
#         """
#         tokens = self.tokenizer.tokenize(text)
#         if len(tokens) > max_length:
#             if head_tokens + tail_tokens > max_length:
#                 raise ValueError("Sum of head_tokens and tail_tokens exceeds max_length.")
#             truncated_tokens = tokens[:head_tokens] + tokens[-tail_tokens:]
#         else:
#             truncated_tokens = tokens
#
#         return self.tokenizer.convert_tokens_to_string(truncated_tokens)
#
#
#     def train_model(self, x_train, y_train, model):
#
#         # Apply truncation consistently
#         self.test_df['truncated_text'] = self.test_df[self.text].apply(
#             lambda x: self.head_tail_truncation(x)
#         )
#         x_eval = self.test_df['truncated_text'].values.tolist()
#         #x_eval = self.test_df[self.text].values.tolist()
#         y_train_binary = self.mlb.fit_transform(y_train)
#
#         X_train = np.array(model.encode(x_train))
#         X_eval = np.array(model.encode(x_eval))
#
#         y_train_binary = np.array(y_train_binary)
#         self.linear_classifier.fit(X_train, y_train_binary)
#         return X_eval, self.linear_classifier, self.mlb
#
#
#     def eval_model(self, X_eval, classifier):
#         """
#         Given the predictions and golds, run the evaluation in several modes: weak, strict, ...
#         :param X_eval: The input features for evaluation.
#         :param classifier: The trained classifier model.
#         :return: A tuple containing various evaluation metrics.
#         """
#
#         threshold = 0.5
#         y_eval = self.test_df[self.labels_col].values.tolist()
#         y_eval_binary = self.mlb.transform(y_eval)
#
#         # Predict probabilities
#         y_pred_proba = classifier.predict_proba(X_eval)
#
#         # Apply threshold to probabilities to obtain binary predictions
#         y_pred = (y_pred_proba >= threshold).astype(int)
#
#         # Compute strong accuracy
#         strict_matches = np.all(y_pred == y_eval_binary, axis=1)
#         strict_accuracy = np.mean(strict_matches)
#
#         # Compute weak accuracy
#         weak_matches = np.any(np.logical_and(y_pred, y_eval_binary), axis=1)
#         weak_accuracy = np.mean(weak_matches)
#
#         # Compute per-class metrics
#         num_classes = y_eval_binary.shape[1]
#         APs = []
#         for j in range(num_classes):
#             AP = average_precision_score(y_eval_binary[:, j], y_pred_proba[:, j])
#             APs.append(AP)
#
#         # Compute macro mAP (unweighted average of APs)
#         macro_mAP = np.mean(APs)
#
#         # Compute weighted mAP (average of APs weighted by number of positives)
#         class_counts = np.sum(y_eval_binary, axis=0)
#         weighted_mAP = np.average(APs, weights=class_counts)
#
#         # Compute micro mAP (global-based)
#         micro_mAP = average_precision_score(y_eval_binary.ravel(), y_pred_proba.ravel())
#
#         # Compute Hamming loss
#         hamming = hamming_loss(y_eval_binary, y_pred)
#
#         # Compute Precision, Recall, F1-Score
#         precision_micro = precision_score(y_eval_binary, y_pred, average='micro')
#         recall_micro = recall_score(y_eval_binary, y_pred, average='micro')
#         precision_macro = precision_score(y_eval_binary, y_pred, average='macro')
#         recall_macro = recall_score(y_eval_binary, y_pred, average='macro')
#         f1_micro = f1_score(y_eval_binary, y_pred, average='micro')
#         f1_macro = f1_score(y_eval_binary, y_pred, average='macro')
#
#         # Compute Jaccard Similarity
#         jaccard_sim = jaccard_score(y_eval_binary, y_pred, average='micro')
#
#         # Compute Log Loss
#         logloss = log_loss(y_eval_binary, y_pred_proba)
#         predictions = []
#         for instance_probas in y_pred_proba:
#             label_prob_dict = {label: proba
#                                for label, proba in zip(self.mlb.classes_, instance_probas)
#                                if proba >= threshold}
#             predictions.append(label_prob_dict)
#         print(predictions)
#         predictions_df = pd.DataFrame({
#             "y_pred": predictions,
#             "y_actual": y_eval
#         })
#
#         # Return all the computed metrics
#         return (strict_accuracy, weak_accuracy, hamming, precision_micro, recall_micro, f1_micro, precision_macro,
#                 recall_macro, f1_macro, jaccard_sim, logloss, macro_mAP, weighted_mAP, micro_mAP, APs), predictions_df



# class LinearClassifierSynthetic:
#
#     def __init__(self, test_df, mlb=None):
#         self.linear_classifier = OneVsRestClassifier(LogisticRegression())
#         self.mlb = mlb if mlb is not None else MultiLabelBinarizer()
#         self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
#         self.test_df = test_df
#         self.text = self.test_df.columns.values[3]
#         self.labels_col = self.test_df.columns.values[2]
#
#     def head_tail_truncation(self, text, max_length=512, head_tokens=128, tail_tokens=382):
#         """
#         Truncates the input text, keeping the first `head_tokens` and the last `tail_tokens` tokens,
#         ensuring the total length does not exceed `max_length`.
#         """
#         # if pd.isna(text):
#         #     text = ""  # default empty string if text is NaN/None
#
#         print(text)
#         tokens = self.tokenizer.tokenize(text)
#         if len(tokens) > max_length:
#             if head_tokens + tail_tokens > max_length:
#                 raise ValueError("Sum of head_tokens and tail_tokens exceeds max_length.")
#             truncated_tokens = tokens[:head_tokens] + tokens[-tail_tokens:]
#         else:
#             truncated_tokens = tokens
#
#         return self.tokenizer.convert_tokens_to_string(truncated_tokens)
#
#
#     def train_model(self, x_train, y_train, model):
#
#         # Apply truncation consistently
#         self.test_df['truncated_text'] = self.test_df[self.text].apply(
#             lambda x: self.head_tail_truncation(x)
#         )
#         x_eval = self.test_df['truncated_text'].values.tolist()
#         y_train_binary = self.mlb.fit_transform(y_train)
#
#         X_train = np.array(model.encode(x_train))
#         X_eval = np.array(model.encode(x_eval))
#
#         y_train_binary = np.array(y_train_binary)
#         self.linear_classifier.fit(X_train, y_train_binary)
#         return X_eval, self.linear_classifier, self.mlb
#
#     def eval_model(self, X_eval, classifier):
#         """
#         Given the predictions and golds, run the evaluation in several modes: weak, strict, ...
#         :param X_eval: The input features for evaluation.
#         :param classifier: The trained classifier model.
#         :return: A tuple containing various evaluation metrics.
#         """
#
#         threshold = 0.6
#         y_eval = self.test_df[self.labels_col].values.tolist()
#         y_eval_binary = self.mlb.transform(y_eval)
#
#         # Predict probabilities
#         y_pred_proba = classifier.predict_proba(X_eval)
#
#         # Apply threshold to probabilities to obtain binary predictions
#         y_pred = (y_pred_proba >= threshold).astype(int)
#
#         # Compute strong accuracy
#         strict_matches = np.all(y_pred == y_eval_binary, axis=1)
#         strict_accuracy = np.mean(strict_matches)
#
#         # Compute weak accuracy
#         weak_matches = np.any(np.logical_and(y_pred, y_eval_binary), axis=1)
#         weak_accuracy = np.mean(weak_matches)
#
#         # Compute per-class metrics
#         num_classes = y_eval_binary.shape[1]
#         APs = []
#         for j in range(num_classes):
#             AP = average_precision_score(y_eval_binary[:, j], y_pred_proba[:, j])
#             APs.append(AP)
#
#         # Compute macro mAP (unweighted average of APs)
#         macro_mAP = np.mean(APs)
#
#         # Compute weighted mAP (average of APs weighted by number of positives)
#         class_counts = np.sum(y_eval_binary, axis=0)
#         weighted_mAP = np.average(APs, weights=class_counts)
#
#         # Compute micro mAP (global-based)
#         micro_mAP = average_precision_score(y_eval_binary.ravel(), y_pred_proba.ravel())
#
#         # Compute Hamming loss
#         hamming = hamming_loss(y_eval_binary, y_pred)
#
#         # Compute Precision, Recall, F1-Score
#         precision_micro = precision_score(y_eval_binary, y_pred, average='micro')
#         recall_micro = recall_score(y_eval_binary, y_pred, average='micro')
#         precision_macro = precision_score(y_eval_binary, y_pred, average='macro')
#         recall_macro = recall_score(y_eval_binary, y_pred, average='macro')
#         f1_micro = f1_score(y_eval_binary, y_pred, average='micro')
#         f1_macro = f1_score(y_eval_binary, y_pred, average='macro')
#
#         # Compute Jaccard Similarity
#         jaccard_sim = jaccard_score(y_eval_binary, y_pred, average='micro')
#
#         # Compute Log Loss
#         logloss = log_loss(y_eval_binary, y_pred_proba)
#         predictions = []
#         for instance_probas in y_pred_proba:
#             label_prob_dict = {label: proba
#                                for label, proba in zip(self.mlb.classes_, instance_probas)
#                                if proba >= threshold}
#             predictions.append(label_prob_dict)
#         print(predictions)
#         predictions_df = pd.DataFrame({
#             "y_pred": predictions,
#             "y_actual": y_eval
#         })
#
#         # Return all the computed metrics
#         return (strict_accuracy, weak_accuracy, hamming, precision_micro, recall_micro, f1_micro, precision_macro,
#                 recall_macro, f1_macro, jaccard_sim, logloss, macro_mAP, weighted_mAP, micro_mAP, APs), predictions_df


class LinearClassifierSynthetic:
    def __init__(self, test_df, mlb=None):
        self.linear_classifier = OneVsRestClassifier(LogisticRegression())
        self.mlb = mlb if mlb is not None else MultiLabelBinarizer()
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
        self.test_df = test_df
        self.text = self.test_df.columns.values[3]  # Ensure index consistency
        self.labels_col = self.test_df.columns.values[2]

    def head_tail_truncation(self, text, max_length=512, head_tokens=128, tail_tokens=382):
        """Truncates text to preserve head and tail tokens."""
        if not isinstance(text, str) or pd.isna(text):
            text = ""
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > max_length:
            truncated_tokens = tokens[:head_tokens] + tokens[-tail_tokens:]
        else:
            truncated_tokens = tokens
        return self.tokenizer.convert_tokens_to_string(truncated_tokens)

    def train_model(self, x_train, y_train, model):
        """Encodes and trains the classifier using training data."""

        # **Ensure y_train is binary transformed**
        y_train_binary = self.mlb.fit_transform(y_train)

        # **Encode x_train (convert text to numerical embeddings)**
        X_train = np.array(model.encode(x_train, convert_to_numpy=True))

        # **Fit the classifier**
        self.linear_classifier.fit(X_train, y_train_binary)

        return self.linear_classifier, self.mlb

    def eval_model(self, model, classifier):
        """Encodes evaluation data and computes metrics."""
        threshold = 0.6
        y_eval = self.test_df[self.labels_col].tolist()
        y_eval_binary = self.mlb.transform(y_eval)

        # **Apply truncation and encode using the model**
        self.test_df['truncated_text'] = self.test_df[self.text].apply(lambda x: self.head_tail_truncation(x))
        x_eval = self.test_df['truncated_text'].tolist()
        X_eval = np.array(model.encode(x_eval, convert_to_numpy=True))

        if X_eval.shape[0] != y_eval_binary.shape[0]:
            raise ValueError(
                f"Mismatch: X_eval has {X_eval.shape[0]} samples, y_eval has {y_eval_binary.shape[0]} samples.")

        # **Predict probabilities and apply threshold**
        y_pred_proba = classifier.predict_proba(X_eval)
        y_pred = (y_pred_proba >= threshold).astype(int)

        return self.compute_metrics(y_eval_binary, y_pred, y_pred_proba, y_eval)

    def compute_metrics(self, y_eval_binary, y_pred, y_pred_proba, y_eval):
        """Computes classification metrics."""
        strict_accuracy = np.mean(np.all(y_pred == y_eval_binary, axis=1))
        weak_accuracy = np.mean(np.any(np.logical_and(y_pred, y_eval_binary), axis=1))
        num_classes = y_eval_binary.shape[1]
        APs = [average_precision_score(y_eval_binary[:, j], y_pred_proba[:, j]) for j in range(num_classes)]
        macro_mAP = np.mean(APs)
        weighted_mAP = np.average(APs, weights=np.sum(y_eval_binary, axis=0))
        micro_mAP = average_precision_score(y_eval_binary.ravel(), y_pred_proba.ravel())
        hamming = hamming_loss(y_eval_binary, y_pred)
        precision_micro = precision_score(y_eval_binary, y_pred, average='micro', zero_division=0)
        recall_micro = recall_score(y_eval_binary, y_pred, average='micro', zero_division=0)
        f1_micro = f1_score(y_eval_binary, y_pred, average='micro', zero_division=0)
        precision_macro = precision_score(y_eval_binary, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_eval_binary, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_eval_binary, y_pred, average='macro', zero_division=0)
        jaccard_sim = jaccard_score(y_eval_binary, y_pred, average='micro', zero_division=0)
        logloss = log_loss(y_eval_binary, y_pred_proba)

        # Format predictions
        # predictions = [{label: proba for label, proba in zip(self.mlb.classes_, instance_probas) if proba >= 0.6}
        #                for instance_probas in y_pred_proba]
        predictions = [[label for label, proba in zip(self.mlb.classes_, instance_probas) if proba >= 0.6]
                       for instance_probas in y_pred_proba]
        predictions_df = pd.DataFrame({"y_pred": predictions, "y_actual": y_eval})

        return (strict_accuracy, weak_accuracy, hamming, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro,
                jaccard_sim, logloss, macro_mAP, weighted_mAP, micro_mAP, APs), predictions_df


class LinearClassifierORO(LinearClassifierSynthetic):
    def __init__(self, test_df, mlb=None):
        super().__init__(test_df, mlb)
        self.text = self.test_df.columns.values[4]  # ORO dataset has different column index for text
        self.labels_col = self.test_df.columns.values[3]  # ORO dataset has different index for labels

    def eval_model(self, model, classifier):
        """Encodes evaluation data and computes metrics specifically for ORO dataset."""
        threshold = 0.6
        y_eval = self.test_df[self.labels_col].tolist()
        y_eval_binary = self.mlb.transform(y_eval)

        # **Apply truncation and encode using the model**
        self.test_df['truncated_text'] = self.test_df[self.text].apply(lambda x: self.head_tail_truncation(x))
        x_eval = self.test_df['truncated_text'].tolist()
        X_eval = np.array(model.encode(x_eval, convert_to_numpy=True))

        if X_eval.shape[0] != y_eval_binary.shape[0]:
            raise ValueError(f"Mismatch: X_eval has {X_eval.shape[0]} samples, y_eval has {y_eval_binary.shape[0]} samples.")

        # **Predict probabilities and apply threshold**
        y_pred_proba = classifier.predict_proba(X_eval)
        y_pred = (y_pred_proba >= threshold).astype(int)

        return self.compute_metrics(y_eval_binary, y_pred, y_pred_proba, y_eval)
# class LinearClassifierORO:
#
#     def __init__(self, test_df, mlb=None):
#         self.linear_classifier = OneVsRestClassifier(LogisticRegression())
#         self.mlb = mlb if mlb is not None else MultiLabelBinarizer()
#         self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
#         self.test_df = test_df
#         self.text = self.test_df.columns.values[4]
#         self.labels_col = self.test_df.columns.values[3]
#
#     def head_tail_truncation(self, text, max_length=512, head_tokens=128, tail_tokens=382):
#         """
#         Truncates the input text, keeping the first `head_tokens` and the last `tail_tokens` tokens,
#         ensuring the total length does not exceed `max_length`.
#         """
#         tokens = self.tokenizer.tokenize(text)
#         if len(tokens) > max_length:
#             if head_tokens + tail_tokens > max_length:
#                 raise ValueError("Sum of head_tokens and tail_tokens exceeds max_length.")
#             truncated_tokens = tokens[:head_tokens] + tokens[-tail_tokens:]
#         else:
#             truncated_tokens = tokens
#
#         return self.tokenizer.convert_tokens_to_string(truncated_tokens)
#
#
#     def train_model(self, x_train, y_train, model):
#         # Apply truncation consistently
#         self.test_df['truncated_text'] = self.test_df[self.text].apply(
#             lambda x: self.head_tail_truncation(x)
#         )
#         x_eval = self.test_df['truncated_text'].values.tolist()
#         #x_eval = self.test_df['combined_text'].values.tolist()
#         y_train_binary = self.mlb.fit_transform(y_train)
#         X_train = np.array(model.encode(x_train))
#         X_eval = np.array(model.encode(x_eval))
#
#         y_train_binary = np.array(y_train_binary)
#         #self.linear_classifier.fit(X_train, y_train)
#         self.linear_classifier.fit(X_train, y_train_binary)
#         return X_eval, self.linear_classifier, self.mlb
#
#
#     def eval_model(self, X_eval, classifier):
#         """
#         Given the predictions and golds, run the evaluation in several modes: weak, strict, ...
#         :param X_eval: The input features for evaluation.
#         :param classifier: The trained classifier model.
#         :return: A tuple containing various evaluation metrics.
#         """
#
#         threshold = 0.6
#         y_eval = self.test_df[self.labels_col].values.tolist()
#         y_eval_binary = self.mlb.transform(y_eval)
#
#         # **Recompute X_eval to match the test dataset**
#         self.test_df['truncated_text'] = self.test_df[self.text].apply(lambda x: self.head_tail_truncation(x))
#         x_eval = self.test_df['truncated_text'].values.tolist()
#         X_eval = np.array(self.model.encode(x_eval))  # Recompute embeddings
#
#         # **Check shape match**
#         if X_eval.shape[0] != y_eval_binary.shape[0]:
#             raise ValueError(
#                 f"Mismatch: X_eval has {X_eval.shape[0]} samples, y_eval has {y_eval_binary.shape[0]} samples.")
#
#         # Predict probabilities
#         y_pred_proba = classifier.predict_proba(X_eval)
#
#         # Apply threshold to probabilities to obtain binary predictions
#         y_pred = (y_pred_proba >= threshold).astype(int)
#
#         # Compute strong accuracy
#         strict_matches = np.all(y_pred == y_eval_binary, axis=1)
#         strict_accuracy = np.mean(strict_matches)
#
#         # Compute weak accuracy
#         weak_matches = np.any(np.logical_and(y_pred, y_eval_binary), axis=1)
#         weak_accuracy = np.mean(weak_matches)
#
#         # Compute per-class metrics
#         num_classes = y_eval_binary.shape[1]
#         APs = []
#         for j in range(num_classes):
#             AP = average_precision_score(y_eval_binary[:, j], y_pred_proba[:, j])
#             APs.append(AP)
#
#         # Compute macro mAP (unweighted average of APs)
#         macro_mAP = np.mean(APs)
#
#         # Compute weighted mAP (average of APs weighted by number of positives)
#         class_counts = np.sum(y_eval_binary, axis=0)
#         weighted_mAP = np.average(APs, weights=class_counts)
#
#         # Compute micro mAP (global-based)
#         micro_mAP = average_precision_score(y_eval_binary.ravel(), y_pred_proba.ravel())
#
#         # Compute Hamming loss
#         hamming = hamming_loss(y_eval_binary, y_pred)
#
#         # Compute Precision, Recall, F1-Score
#         precision_micro = precision_score(y_eval_binary, y_pred, average='micro')
#         recall_micro = recall_score(y_eval_binary, y_pred, average='micro')
#         precision_macro = precision_score(y_eval_binary, y_pred, average='macro')
#         recall_macro = recall_score(y_eval_binary, y_pred, average='macro')
#         f1_micro = f1_score(y_eval_binary, y_pred, average='micro')
#         f1_macro = f1_score(y_eval_binary, y_pred, average='macro')
#
#         # Compute Jaccard Similarity
#         jaccard_sim = jaccard_score(y_eval_binary, y_pred, average='micro')
#
#         # Compute Log Loss
#         logloss = log_loss(y_eval_binary, y_pred_proba)
#         # Save predictions and actual values
#         # Generating predictions with probabilities clearly
#         predictions = []
#         for instance_probas in y_pred_proba:
#             label_prob_dict = {label: proba
#                                for label, proba in zip(self.mlb.classes_, instance_probas)
#                                if proba >= threshold}
#             predictions.append(label_prob_dict)
#         print(predictions)
#         predictions_df = pd.DataFrame({
#             "y_pred": predictions,
#             "y_actual": y_eval
#         })
#         #predictions_df.to_csv("predictions_.tsv", sep='\t', index=False)
#
#         # Return all the computed metrics
#         return (strict_accuracy, weak_accuracy, hamming, precision_micro, recall_micro, f1_micro, precision_macro,
#                 recall_macro, f1_macro, jaccard_sim, logloss, macro_mAP, weighted_mAP, micro_mAP, APs), predictions_df
#
#
#     def evaluate_metrics(self, y_eval, y_pred, y_pred_proba):
#         # Handle empty values (consider filtering or imputation)
#         valid_indices = [i for i, row in enumerate(y_eval) if any(row)]  # Indices with non-empty labels
#         y_eval = np.array(y_eval)[valid_indices]
#         y_pred = y_pred[valid_indices]
#         y_pred_proba = y_pred_proba[valid_indices]
#
#         # Multi-label binarization (assuming self.mlb is already fit)
#         y_eval_binary = self.mlb.transform(y_eval)
#
#         # Compute metrics
#         strict_matches = np.all(y_pred == y_eval_binary, axis=1)
#         strict_accuracy = np.mean(strict_matches)
#
#         weak_matches = np.any(np.logical_and(y_pred, y_eval_binary), axis=1)
#         weak_accuracy = np.mean(weak_matches)
#
#         num_classes = y_eval_binary.shape[1]
#         APs = []
#         class_counts = np.zeros(num_classes)  # Initialize class counts with zeros
#         for j in range(num_classes):
#             # Check for empty class in ground truth
#             if not np.any(y_eval_binary[:, j]):
#                 APs.append(0)  # Assign 0 AP for empty class
#             else:
#                 # Calculate AP for non-empty class
#                 AP = average_precision_score(y_eval_binary[:, j], y_pred_proba[:, j])
#                 APs.append(AP)
#                 class_counts[j] = np.sum(y_eval_binary[:, j])  # Update class count
#
#         # Compute mAPs
#         macro_mAP = np.mean(APs)
#         weighted_mAP = np.average(APs, weights=class_counts)
#         micro_mAP = average_precision_score(y_eval_binary.ravel(), y_pred_proba.ravel())
#
#         # Other metrics
#         hamming = hamming_loss(y_eval_binary, y_pred)
#         precision_micro = precision_score(y_eval_binary, y_pred, average='micro')
#         recall_micro = recall_score(y_eval_binary, y_pred, average='micro')
#         f1_micro = f1_score(y_eval_binary, y_pred, average='micro')
#         precision_macro = precision_score(y_eval_binary, y_pred, average='macro')
#         recall_macro = recall_score(y_eval_binary, y_pred, average='macro')
#         f1_macro = f1_score(y_eval_binary, y_pred, average='macro')
#         jaccard_sim = jaccard_score(y_eval_binary, y_pred, average='micro')
#         logloss = log_loss(y_eval_binary, y_pred_proba)
#
#         # Return dictionary of metrics
#         metrics = {
#             "Strict Accuracy": strict_accuracy,
#             "Weak Accuracy": weak_accuracy,
#             "Macro mAP": macro_mAP,
#             "Weighted mAP": weighted_mAP,
#             "Micro mAP": micro_mAP,
#             "Hamming Loss": hamming,
#             "Precision (Micro)": precision_micro,
#             "Recall (Micro)": recall_micro,
#             "F1 Score (Micro)": f1_micro,
#             "Precision (Macro)": precision_macro,
#             "Recall (Macro)": recall_macro,
#             "F1 Score (Macro)": f1_macro,
#             "Jaccard Similarity": jaccard_sim,
#             "Log Loss": logloss
#         }
#
#         return metrics


class Predict:

    def __init__(self, linear_classifier, embedding_model, mlb):

        #self.oro_df = oro_df
        self.linear_classifier = linear_classifier
        self.embedding_model = embedding_model
        self.mlb = mlb
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
        self.threshold = 0.6


    def head_tail_truncation(self, text, max_length=512, head_tokens=128, tail_tokens=382):
        """
        Truncates the input text, keeping the first `head_tokens` and the last `tail_tokens` tokens,
        ensuring the total length does not exceed `max_length`.
        """
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > max_length:
            if head_tokens + tail_tokens > max_length:
                raise ValueError("Sum of head_tokens and tail_tokens exceeds max_length.")
            truncated_tokens = tokens[:head_tokens] + tokens[-tail_tokens:]
        else:
            truncated_tokens = tokens

        return self.tokenizer.convert_tokens_to_string(truncated_tokens)

    def predict_from_file(self, oro_df):

        oro_df['truncated_text'] = oro_df['text'].apply(lambda x: self.head_tail_truncation(x))
        x_eval_truncated = oro_df['truncated_text'].values.tolist()
        X_eval = np.array(self.embedding_model.encode(x_eval_truncated))
        y_pred_proba = self.linear_classifier.predict_proba(X_eval)

        # Step 4: Apply threshold to get binary predictions
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        # Step 5: Convert binary predictions back to labels
        predicted_labels = self.mlb.inverse_transform(y_pred)

        # Convert y_pred_proba to a list of dictionaries with label: probability for each instance
        predicted_probs = []
        for i in range(len(y_pred_proba)):
            label_prob_dict = {}
            for label, proba in zip(self.mlb.classes_, y_pred_proba[i]):
                if proba >= self.threshold:
                    label_prob_dict[label] = proba
            predicted_probs.append(label_prob_dict)

        return predicted_probs

    def predict_from_title_abstract(self, title, abstract):

        title = title if title is not None else ""
        abstract = abstract if abstract is not None else ""
        text = title + ". " + abstract
        truncated_text = self.head_tail_truncation(text)
        X_eval = np.array(self.embedding_model.encode([truncated_text]))
        y_pred_proba = self.linear_classifier.predict_proba(X_eval)[0]

        label_prob_dict = {label: proba for label, proba in zip(self.mlb.classes_, y_pred_proba) if
                           proba >= self.threshold}

        return label_prob_dict

    def predict_from_fulltext(self, fulltext):
        text = fulltext
        truncated_text = self.head_tail_truncation(text)
        X_eval = np.array(self.embedding_model.encode([truncated_text]))
        y_pred_proba = self.linear_classifier.predict_proba(X_eval)[0]

        label_prob_dict = {label: proba for label, proba in zip(self.mlb.classes_, y_pred_proba) if
                           proba >= self.threshold}

        return label_prob_dict

    # def predict_from_coreid(self, core_id):
    #     """
    #     Predict SDGs from a core ID by retrieving the corresponding text.
    #     """
    #     row = self.oro_df[self.oro_df['id'] == core_id]
    #     if row.empty:
    #         return {"error": "Core ID not found."}
    #
    #     title = row['title'].values[0]
    #     abstract = row['abstract'].values[0]
    #
    #     return self.predict_from_title_abstract(title, abstract)



def desc_finetuning(model):
    LABEL_DESC_DIR = os.path.join(DATA_DIR, 'label_desc')

    sdg_data_loader = DescriptionDataLoader(LABEL_DESC_DIR, model)
    sdg_trainer = DescriptionFineTuning(sdg_data_loader.sdg_definitions, NON_OVERLAPPING_SDGS, model)

    train_examples = sdg_trainer.prepare_data()
    model = sdg_trainer.train_model(train_examples)
    return model


def multi_label_trainer(model):
    args = get_args()
    trained_model_dir = str
    config_data = load_config()
    logger = logging.getLogger(__name__)

    SEED = args.seed if hasattr(args, "seed") else 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if "timed_dir" in config_data:
        trained_model_dir = config_data["timed_dir"]
    else:
        logger.info('Check the config path')

    dataset_type = args.dataset
    if dataset_type not in ['knowledge_hub', 'synthetic']:
        raise ValueError('Supported dataset types: knowledge_hub, synthetic')

    # **Load dataset and fine-tuning classes dynamically**
    data_loader_class = LOADER_CLASSES[dataset_type]
    finetuning_class = FINETUNING_CLASSES[dataset_type]
    multi_label_data_loader = data_loader_class(MULTI_LABEL_DATA_DIRS[dataset_type])
    train_df, default_test_df = multi_label_data_loader.read_dataset()

    # **Override test_df if in-domain evaluation is requested**
    if args.do_in_domain_eval:
        logger.info("Reading in-domain evaluation dataset")
        oro_loader = LOADER_CLASSES['oro'](MULTI_LABEL_DATA_DIRS['oro'])
        test_df = oro_loader.read_dataset()
    else:
        test_df = default_test_df

    # **Fine-tuning**
    SBERT_finetuning = finetuning_class(train_df, model)
    x_train, y_train, train_examples = SBERT_finetuning.prepare_data()

    if args.multi_label_finetuning:
        embedding_model = SBERT_finetuning.sbert_finetuning(train_examples)
    else:
        embedding_model = model

    # **Save embedding model**
    multi_label_model_path = os.path.join(MODEL_DIR, os.path.basename(trained_model_dir))
    embedding_model.save_pretrained(os.path.join(multi_label_model_path, f"sbert_embedding_model_{dataset_type}"))

    if dataset_type not in TRAINER_CLASSES:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Expected 'synthetic' or 'knowledge_hub'.")

    trainer_class = TRAINER_CLASSES[dataset_type]
    multilabel_sdg_trainer = trainer_class(train_df)

    classifier, mlb = multilabel_sdg_trainer.train_model(x_train, y_train, embedding_model)

    #Save classifier and MultiLabelBinarizer
    with open(os.path.join(multi_label_model_path, f"linear_classifier_{dataset_type}.pkl"), "wb") as model_file:
        pickle.dump(classifier, model_file)

    with open(os.path.join(multi_label_model_path, f"mlb_{dataset_type}.pkl"), "wb") as mlb_file:
        pickle.dump(mlb, mlb_file)

    # **Evaluation: Use ORO Dataset if Required**
    if args.do_in_domain_eval:
        logger.info("Performing in-domain evaluation using ORO dataset")
        evaluation_trainer = LinearClassifierORO(test_df, mlb)  # Use gold dataset
    else:
        if dataset_type == "knowledge_hub":
            logger.info("Evaluating using Knowledge Hub dataset")
            evaluation_trainer = LinearClassifierKnowledgeHub(test_df, mlb)
        elif dataset_type == "synthetic":
            logger.info("Evaluating using Synthetic dataset")
            evaluation_trainer = LinearClassifierSynthetic(test_df, mlb)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        #evaluation_trainer = LinearClassifierSynthetic(test_df, mlb)  # Use same dataset as training

    # **Pass `embedding_model` to `eval_model()` to ensure correct embeddings**
    results, predictions = evaluation_trainer.eval_model(embedding_model, classifier)

    return results, predictions

# def multi_label_trainer(model):
#     args = get_args()
#     trained_model_dir = str
#     config_data = load_config()
#     logger = logging.getLogger(__name__)
#
#     if "timed_dir" in config_data:
#         trained_model_dir = config_data["timed_dir"]
#     else:
#         logger.info('Check the config path')
#
#     dataset_type = args.dataset
#     if dataset_type not in ['knowledge_hub', 'synthetic']:
#         raise ValueError('Supported dataset types: knowledge_hub, synthetic')
#
#     # Load dataset and fine-tuning classes dynamically
#     data_loader_class = LOADER_CLASSES[dataset_type]
#     finetuning_class = FINETUNING_CLASSES[dataset_type]
#     multi_label_data_loader = data_loader_class(MULTI_LABEL_DATA_DIRS[dataset_type])
#     train_df, default_test_df = multi_label_data_loader.read_dataset()
#
#     # Override test_df if in-domain evaluation is requested
#     if args.do_in_domain_eval:
#         logger.info("Reading in-domain evaluation dataset")
#         oro_loader = LOADER_CLASSES['oro'](MULTI_LABEL_DATA_DIRS['oro'])
#         test_df = oro_loader.read_dataset()
#     else:
#         test_df = default_test_df
#
#     # Fine-tuning
#     SBERT_finetuning = finetuning_class(train_df, model)
#     x_train, y_train, train_examples = SBERT_finetuning.prepare_data()
#
#     if args.multi_label_finetuning:
#         embedding_model = SBERT_finetuning.sbert_finetuning(train_examples)
#     else:
#         embedding_model = model
#
#     # Save embedding model
#     multi_label_model_path = os.path.join(MODEL_DIR, os.path.basename(trained_model_dir))
#     embedding_model.save_pretrained(os.path.join(multi_label_model_path, f"sbert_embedding_model_{dataset_type}"))
#
#     # **Train on Synthetic Data Only**
#     trainer_class = LinearClassifierSynthetic
#     multilabel_sdg_trainer = trainer_class(train_df)
#
#     classifier, mlb = multilabel_sdg_trainer.train_model(x_train, y_train)
#
#     # Save classifier and MultiLabelBinarizer
#     with open(os.path.join(multi_label_model_path, f"linear_classifier_{dataset_type}.pkl"), "wb") as model_file:
#         pickle.dump(classifier, model_file)
#
#     with open(os.path.join(multi_label_model_path, f"mlb_{dataset_type}.pkl"), "wb") as mlb_file:
#         pickle.dump(mlb, mlb_file)
#
#     # **Evaluation: Use ORO Dataset if Required**
#     if args.do_in_domain_eval:
#         logger.info("Performing in-domain evaluation using ORO dataset")
#         evaluation_trainer = LinearClassifierORO(test_df, mlb)  # Use gold dataset
#     else:
#         evaluation_trainer = LinearClassifierSynthetic(test_df, mlb)  # Use same dataset as training
#
#     # Pass `model` to `eval_model()` to ensure embeddings are computed correctly
#     results, predictions = evaluation_trainer.eval_model(embedding_model, classifier)
#
#     return results, predictions

# def multi_label_trainer(model):
#     args = get_args()
#     trained_model_dir = str
#     config_data = load_config()
#     logger = logging.getLogger(__name__)
#     if "timed_dir" in config_data:
#         trained_model_dir = config_data["timed_dir"]
#     else:
#         logger.info('Check the config path')
#
#     dataset_type = args.dataset
#     if dataset_type not in ['knowledge_hub', 'synthetic']:
#         raise ValueError('Supported dataset types: knowledge_hub, synthetic')
#
#     # Load dataset and fine-tuning classes dynamically
#     data_loader_class = LOADER_CLASSES[dataset_type]
#     finetuning_class = FINETUNING_CLASSES[dataset_type]
#     multi_label_data_loader = data_loader_class(MULTI_LABEL_DATA_DIRS[dataset_type])
#     train_df, default_test_df = multi_label_data_loader.read_dataset()
#
#     # Override test_df if in-domain evaluation is requested
#     if args.do_in_domain_eval:
#         logger.info("Reading in-domain evaluation dataset")
#         oro_loader = LOADER_CLASSES['oro'](MULTI_LABEL_DATA_DIRS['oro'])
#         test_df = oro_loader.read_dataset()
#     else:
#         test_df = default_test_df
#
#     # Fine-tuning
#     SBERT_finetuning = finetuning_class(train_df, model)
#     x_train, y_train, train_examples = SBERT_finetuning.prepare_data()
#
#     if args.multi_label_finetuning:
#         embedding_model = SBERT_finetuning.sbert_finetuning(train_examples)
#     else:
#         embedding_model = model
#
#     # Save embedding model
#     multi_label_model_path = os.path.join(MODEL_DIR, os.path.basename(trained_model_dir))
#     embedding_model.save_pretrained(os.path.join(multi_label_model_path, f"sbert_embedding_model_{dataset_type}"))
#
#     # **Train on Synthetic Data Only**
#     trainer_class = LinearClassifierSynthetic
#     multilabel_sdg_trainer = trainer_class(train_df)
#
#     classifier, mlb = multilabel_sdg_trainer.train_model(x_train, y_train)
#
#     # Save classifier and MultiLabelBinarizer
#     with open(os.path.join(multi_label_model_path, f"linear_classifier_{dataset_type}.pkl"), "wb") as model_file:
#         pickle.dump(classifier, model_file)
#
#     with open(os.path.join(multi_label_model_path, f"mlb_{dataset_type}.pkl"), "wb") as mlb_file:
#         pickle.dump(mlb, mlb_file)
#
#     # **Evaluation: Use ORO Dataset if Required**
#     if args.do_in_domain_eval:
#         logger.info("Performing in-domain evaluation using ORO dataset")
#         evaluation_trainer = LinearClassifierORO(test_df, mlb)  # Use gold dataset
#     else:
#         evaluation_trainer = LinearClassifierSynthetic(test_df, mlb)  # Use same dataset
#
#     results, predictions = evaluation_trainer.eval_model(X_eval, classifier)
#
#     return results, predictions


def sdg_prediction(linear_classifier, embedding_model, mlb):
    METADATA_DIR = os.path.join(OUTPUT_DIR, 'metadata')
    oro_data_loader = ORODataLoader(METADATA_DIR)
    oro_df = oro_data_loader.read_dataset()
    core_ids = oro_df['id']
    inference = Predict(linear_classifier, embedding_model, mlb)
    predicted_probs = inference.predict_from_file(oro_df)
    # predicted_labels_str = [', '.join(labels) for labels in predicted_labels]

    results = []
    for idx, prob_dict in enumerate(predicted_probs):
        if not prob_dict:  # Check if the dictionary is empty
            result = {
                "id": core_ids.iloc[idx],
                "predictions": None,  # or use a placeholder value like "No Prediction"
                "confidence_score": None  # or a placeholder value like 0
            }
            results.append(result)
        else:
            predictions = list(prob_dict.keys())
            confidence_scores = list(prob_dict.values())

            # Split the predictions and confidence scores
            for pred, conf in zip(predictions, confidence_scores):
                result = {
                    "id": core_ids.iloc[idx],
                    "predictions": pred,
                    "confidence_score": round(conf * 100, 2)
                }
                results.append(result)

    return results

def sdg_prediction_app(linear_classifier, embedding_model, mlb, input_type, input_value):

    results = []
    inference = Predict(linear_classifier, embedding_model, mlb)
    # Handle different input types
    if input_type == 'file':
        file_path = input_value
        data_loader = DatasetLoader(file_path)
        df = data_loader.read_dataset()

        # Ensure 'date' is a string and clean it
        df['date'] = df['date'].astype(str)
        df['date'] = df['date'].str.strip()

        def parse_date(date_str):
            if pd.isnull(date_str) or date_str.strip() == '':
                return pd.NaT
            date_str = date_str.strip()
            for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except ValueError:
                    continue
            return pd.NaT

        # Apply the custom date parsing function
        df['date_parsed'] = df['date'].apply(parse_date)
        df['year'] = df['date_parsed'].dt.year

        core_ids = df['id']
        years = df['year']
        predicted_probs = inference.predict_from_file(df)
        for idx, prob_dict in enumerate(predicted_probs):
            year = years.iloc[idx]
            if pd.isnull(year):
                year = None
            else:
                year = int(year)
            if not prob_dict:  # Check if the dictionary is empty
                result = {
                    "id": core_ids.iloc[idx],
                    "predictions": None,  # or use a placeholder value like "No Prediction"
                    "confidence_score": None,  # or a placeholder value like 0
                    "year": year
                }
                results.append(result)
            else:
                predictions = list(prob_dict.keys())
                confidence_scores = list(prob_dict.values())

                # Split the predictions and confidence scores
                for pred, conf in zip(predictions, confidence_scores):
                    result = {
                        "id": core_ids.iloc[idx],
                        "predictions": pred,
                        "confidence_score": round(conf * 100, 2),
                        "year": year
                    }
                    results.append(result)

    elif input_type == 'text':
        title, abstract = input_value
        response = query_es_by_title(title)
        core_id = get_core_id(response)
        prob_dict = inference.predict_from_title_abstract(title, abstract)
        if not prob_dict:  # Check if the dictionary is empty
            result = {
                "id": core_id,
                "predictions": None,  # or use a placeholder value like "No Prediction"
                "confidence_score": None  # or a placeholder value like 0
            }
            results.append(result)
        else:
            for pred, conf in prob_dict.items():
                result = {
                    "id": core_id,
                    "predictions": pred,
                    "confidence_score": round(conf * 100, 2)
                }
                results.append(result)
    elif input_type == 'fulltext':
        text = input_value
        prob_dict = inference.predict_from_fulltext(text)
        if not prob_dict:  # Check if the dictionary is empty
            result = {
                "predictions": None,  # or use a placeholder value like "No Prediction"
                "confidence_score": None  # or a placeholder value like 0
            }
            results.append(result)
        else:
            for pred, conf in prob_dict.items():
                result = {
                    "predictions": pred,
                    "confidence_score": round(conf * 100, 2)
                }
                results.append(result)
    elif input_type == 'core_id':
        core_id = input_value
        response = query_es_by_id(core_id)
        metadata_processor = CORESingleMetaDataExtraction()
        title_abstract = metadata_processor.get_title_abstract_es(response)
        title = title_abstract.get('title', '')
        abstract = title_abstract.get('abstract', '')
        prob_dict = inference.predict_from_title_abstract(title, abstract)
        if not prob_dict:  # Check if the dictionary is empty
            result = {
                "id": core_id,
                "predictions": None,  # or use a placeholder value like "No Prediction"
                "confidence_score": None  # or a placeholder value like 0
            }
            results.append(result)

        else:
            for pred, conf in prob_dict.items():
                result = {
                    "id": core_id,
                    "predictions": pred,
                    "confidence_score": round(conf * 100, 2)
                }
                results.append(result)

    elif input_type == 'entities_bulk':
        """
        input_value is a list of dictionaries:
        [
          { "id": 101, "title": "...", "description": "..." },
          { "id": 102, "title": "...", "description": "..." },
          ...
        ]
        We'll build a pandas DataFrame in-memory, then reuse .predict_from_file style logic.
        """

        entities_list = input_value
        rows = []
        for e in entities_list:
            rows.append({
                "id": e["id"],
                "title": e["title"] or "",
                "abstract": e["description"] or ""
            })
        df = pd.DataFrame(rows)
        df["title"].fillna("", inplace=True)
        df["abstract"].fillna("", inplace=True)
        df["text"] = df["title"] + ". " + df["abstract"]

        predicted_probs = inference.predict_from_file(df)
        core_ids = df["id"].values
        for idx, prob_dict in enumerate(predicted_probs):
            doc_id = core_ids[idx]
            add_prob_dict_to_response(doc_id, prob_dict, results)

    if input_type == 'single_doc':
        """
        input_value is a tuple (doc_id, title, description).
        We'll do something similar to 'predict_from_title_abstract'.
        """
        doc_id, title, description = input_value
        # Ensure they are not None
        title = title if title else ""
        description = description if description else ""

        # Use your 'predict_from_title_abstract' or do it inline:
        prob_dict = inference.predict_from_title_abstract(title, description)

        add_prob_dict_to_response(doc_id, prob_dict, results)

    return results



def add_prob_dict_to_response(doc_id, prob_dict, results):
    if not prob_dict:
        results.append({
            "id": doc_id,
            "predictions": None,
            "confidence_score": None
        })
    else:
        for pred_label, conf in prob_dict.items():
            results.append({
                "id": doc_id,
                "predictions": pred_label,
                "confidence_score": round(conf * 100, 2)
            })

# def sdg_prediction_app(linear_classifier, embedding_model, mlb, input_type, input_value):
#
#     results = []
#     inference = Predict(linear_classifier, embedding_model, mlb)
#     # Handle different input types
#     if input_type == 'file':
#         file_path = input_value
#         data_loader = DataLoader(file_path)
#         df = data_loader.read_dataset()
#         core_ids = df['id']
#         predicted_probs = inference.predict_from_file(df)
#         for idx, prob_dict in enumerate(predicted_probs):
#             predictions = list(prob_dict.keys())
#             confidence_scores = list(prob_dict.values())
#
#             # Split the predictions and confidence scores
#             for pred, conf in zip(predictions, confidence_scores):
#                 result = {
#                     "id": core_ids.iloc[idx],
#                     "predictions": pred,
#                     "confidence_score": round(conf * 100, 2)
#                 }
#                 results.append(result)
#
#     elif input_type == 'text':
#         title, abstract = input_value
#         response = query_es_by_title(title)
#         core_id = get_core_id(response)
#         prob_dict = inference.predict_from_title_abstract(title, abstract)
#
#         for pred, conf in prob_dict.items():
#             result = {
#                 "id": core_id,
#                 "predictions": pred,
#                 "confidence_score": round(conf * 100, 2)
#             }
#             results.append(result)
#
#     elif input_type == 'core_id':
#         core_id = input_value
#         response = query_es_by_id(core_id)
#         metadata_processor = CORESingleMetaDataExtraction()
#         title_abstract = metadata_processor.get_title_abstract_es(response)
#         title = title_abstract.get('title', '')
#         abstract = title_abstract.get('abstract', '')
#         prob_dict = inference.predict_from_title_abstract(title, abstract)
#
#         # if "error" in prob_dict:
#         #     return prob_dict  # Return the error message if core ID is not found
#
#         for pred, conf in prob_dict.items():
#             result = {
#                 "id": core_id,
#                 "predictions": pred,
#                 "confidence_score": round(conf * 100, 2)
#             }
#             results.append(result)
#
#     return results




MULTI_LABEL_DATA_DIRS = {
    'knowledge_hub': os.path.join(DATA_DIR, 'sdg_knowledge_hub'),
    'synthetic': os.path.join(DATA_DIR, 'synthetic_data'),
    'oro': os.path.join(DATA_DIR, 'manually_annotated_oro')
}

# Loader and Fine-tuning Class mappings
LOADER_CLASSES = {
    'knowledge_hub': MultiLabelDatasetLoader,
    'synthetic': MultiLabelSyntheticDatasetLoader,
    'oro': MultiLabelDatasetOROLoader
}

FINETUNING_CLASSES = {
    'knowledge_hub': MultiLabelSBERTFineTuning,
    'synthetic': MultiLabelSyntheticSBERTFineTuning,
    'oro': MultiLabelOROSBERTFineTuning
}

TRAINER_CLASSES = {
    'knowledge_hub': LinearClassifierKnowledgeHub,
    'synthetic': LinearClassifierSynthetic
}








