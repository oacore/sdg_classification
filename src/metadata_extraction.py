from tqdm import tqdm
import re
import os
from data import DATA_DIR
from output import OUTPUT_DIR
from es_core import *
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

import requests
from ratelimit import limits, sleep_and_retry

RATE_LIMIT = 100
#apikey="OzdvV26qYimgh4aSL1u5DtjR9WJTZXpG"
apikey = 'test-1b6710afd0c64050a0f16582899'
# CALLS = 30
# RATE_LIMIT = 60


class COREMetaDataExtraction:
    def __init__(self, data_dir, output_dir, core_ids_file):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.core_ids_file = core_ids_file
        self.title_abstract = {}
        self.title_abstracts = []
        self.core_ids = []

    def get_title_abstract_es(self, response):

        try:
            if response['hits']['total'] > 0:
                for hit in response["hits"]["hits"]:
                    title = hit['_source'].get('title', '')
                    if title is not None:
                        title = title.replace("\n", "")
                        title = self.remove_html_tags(title)

                    abstract = hit['_source'].get('description', '')
                    if abstract is not None:
                        abstract = abstract.replace("\n", "")
                        abstract = self.remove_html_tags(abstract)

                    if title or abstract:
                        self.title_abstract['title'] = title
                        self.title_abstract['abstract'] = abstract
                    else:
                        self.title_abstract['title'] = ''
                        self.title_abstract['abstract'] = ''
            else:
                self.title_abstract['title'] = ''
                self.title_abstract['abstract'] = ''

        except Exception as e:
            print("Exception:", e)
            self.title_abstract['title'] = ''
            self.title_abstract['abstract'] = ''

        return self.title_abstract

    @limits(calls=50, period=RATE_LIMIT)
    def get_title_abstract_api(self, id):
        id = str(id)
        headers = {"Authorization": "Bearer " + apikey}
        try:
            response = requests.get("https://api.core.ac.uk/v3/outputs/" + id, params={'id': id},
                                    headers=headers).json()
            title = response['title'].replace("\n", "")
            abstract = response['abstract'].replace("\n", "")
        except ValueError:
            print('No Response')
            title = None
            abstract = None

        return title, abstract

    def remove_html_tags(self, text):
        """Remove HTML tags from a string using regex."""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    def read_clean_file(self, file_path, expected_columns=3):
        rows = []
        with open(file_path, 'r', encoding='utf-8') as file:
            header = file.readline().strip().split('\t')
            for line in file:
                fields = line.strip().split('\t')
                if len(fields) == expected_columns:
                    fields[1] = self.remove_html_tags(fields[1])
                    fields[2] = self.remove_html_tags(fields[2])
                    rows.append(fields)
                else:
                    print(f"Skipping malformed line in {file_path}: {line.strip()}")
        return pd.DataFrame(rows, columns=header)

    def combine_files(self, output_file):
        df_list = []
        for i in range(1, 11):
            file_path = os.path.join(self.data_dir, f"oro_title_abstracts_{i}.txt")
            df = self.read_clean_file(file_path)
            df_list.append(df)

        # Concatenate all DataFrames
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df.to_csv(os.path.join(self.data_dir, output_file), sep='\t', index=False, encoding='utf-8')

    def process_core_ids(self):
        CORE_IDs_DIR = os.path.join(self.data_dir, 'core_ids')
        if not os.path.exists(CORE_IDs_DIR):
            os.makedirs(CORE_IDs_DIR)
        df = pd.read_csv(self.core_ids_file, sep='\t', encoding='utf-8', engine='python')
        core_id = df['id']

        for id in tqdm(core_id[:500], desc='Processing entries', unit='id'):
            self.core_ids.append(id)
            response = query_es_by_id(id)
            title_abstract_dict = self.get_title_abstract_es(response)
            title = title_abstract_dict.get('title', '')
            abstract = title_abstract_dict.get('abstract', '')
            if title or abstract:
                self.title_abstracts.append((id, title, abstract))

    def save_to_file(self):
        METADATA_DIR = os.path.join(self.output_dir, 'metadata')
        if not os.path.exists(METADATA_DIR):
            os.makedirs(METADATA_DIR)

        with open(os.path.join(METADATA_DIR, "oro_title_abstracts.txt"), 'w', encoding='utf-8') as f:
            f.write("id\ttitle\tabstract\n")  # Column headers
            for entry in self.title_abstracts:
                f.write(f"{entry[0]}\t{entry[1]}\t{entry[2]}\n")

    def run(self):
        self.process_core_ids()
        self.save_to_file()

if __name__ == "__main__":
    # Initialize the MetaData class with appropriate paths
    metadata_processor = COREMetaDataExtraction(DATA_DIR, OUTPUT_DIR, os.path.join(DATA_DIR, 'core_ids', "oro.tsv"))
    metadata_processor.run()







