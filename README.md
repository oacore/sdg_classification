# Multi-label SDG Classification

Install the python dependencies inside a virtual env

````
cd sdg_classification
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
````

## Train a multi-label SDG classifier
### Multi-label SBERT fine-tuning + Classification on synthetic dataset 
````
python3 "$PROJECT_DIR/src/multi_label_sdg.py" --multi_label_finetuning --dataset=synthetic --do_train
````
### Label Description SBERT fine-tuning + Classification on synthetic dataset
````
python3 "$PROJECT_DIR/src/multi_label_sdg.py" --label_desc_finetuning --dataset=synthetic --do_train
````
### Two-stage SBERT fine-tuning + Classification
````
python3 "$PROJECT_DIR/src/multi_label_sdg.py" --multi_label_finetuning --dataset=synthetic --do_train

````
Synthetic dataset is available at data/synthetic_data/synthetic_final.tsv

To train the model on Out-if-Domain (OOD) [Knowledge Hub Dataset](https://zenodo.org/records/7523032),
````
python3 "$PROJECT_DIR/src/multi_label_sdg.py" --multi_label_finetuning --dataset=knowledge_hub --do_train

````
To perform evaluation on the manually annotated multi-label scientific SDG dataset, 
````
python3 "$PROJECT_DIR/src/multi_label_sdg.py" --multi_label_finetuning --dataset=synthetic --do_train --do_in_domain_eval

````

The source code for SBERT fine-tuning and linear classification is largely inspired from [SetFit](https://colab.research.google.com/github/MosheWasserb/SetFit/blob/main/SetFit_SST_2.ipynb#scrollTo=aFOzlLAfYOHU)

### Manually annotated Multi-label SDG dataset
Manually annotated dataset of papers from Open Research Online (ORO) is available at [data/manually_annotated_oro/oro_gold_dataset.txt](https://github.com/oacore/sdg_classification/tree/main/data/manually_annotated_oro/oro_gold_dataset.txt) (final version)



## Running the App
To get results for the input file (containing core_id, title and abstract), or a single title and abstract, or for a CORE_ID, use either of the following steps:

For input file containing metadata - 
````
curl -X GET "http://localhost:5007/classify_file?file_path={path_to_metadata_file}
````
Supported file format - .txt tab seperated file with header: 'id', 'title', 'abstract' and 'date'

For input title and abstract - 
````
curl -X GET "http://localhost:5007/classify_text?title={title}&abstract={abstract}
````

For input core_id - 
````
curl -X GET "http://localhost:5007/classify_coreid?core_id={core_id}
````
where core_id is the identifier of research papers in [https://core.ac.uk/services/api](https://core.ac.uk/services/api) open-access repository    

To get the prediction for a sample file (see output/metadata/oro_title_abstract.txt), run the scripts
````
./run_scripts.sh
````

## Demo page
The source code for the demo page, [CORE Labs](https://core.ac.uk/labs/sdg) is available here -
````
https://github.com/oacore/about
````