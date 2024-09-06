# Multi-label SDG Classification

Install the python dependencies inside a virtual env

````
cd sdg_classification
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
````

To run the scripts
````
./run_scripts.sh
````

## Running the App

For input file containing metadata - 
````
curl -X GET "http://localhost:5007/classify_file?file_path={path_to_metadata_file}
````
Supported file format - .txt tab seperated file with header: 'id', 'title' and 'abstract'

For input title and abstract - 
````
curl -X GET "http://localhost:5007/classify_text?title={title}&abstract={abstract}
````

For input core_id - 
````
curl -X GET "http://localhost:5007/classify_coreid?core_id={core_id}
````
