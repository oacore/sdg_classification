import requests
import json



# dictionary structured like an Elasticsearch query:
QUERY_BY_TITLE = {
  "query": {
    "bool": {
      "must": {
        "match": {
          "title": ""
        }
      }
    }
  },
    "size":5
}

QUERY_BY_REPO_ID = {
  "query": {
    "bool": {
      "must": [
        {
          "term": {
            "deleted": "ALLOWED"
          }
        },
        {
          "nested": {
            "path": "repositories",
            "query": {
              "term": {
                "repositories.id": " "
              }
            }
          }
        }
      ]
    }
  }
}

QUERY_BY_ID = {
  "query": {
    "match": {
      "id": " "
    }
  }
}

SCROLL_QUERY = {
    "scroll": "1m",
    "scroll_id": ""
}

#SEARCH_URL = "http://core-indx-fcr01.open.ac.uk:9200/works/_search"
SCROLL_URL = "http://core-indx-fcr01.open.ac.uk:9200/_search/scroll"


SEARCH_URL = "http://core-indx-fcr01.open.ac.uk:9200/articles/_search"
REPO_SEARCH_URL = "http://core-indx-fcr01.open.ac.uk:9200/repositories/_search"


def query_es_by_id(id):
    headers={"Content-Type":"application/json"}
    #QUERY_BY_ID["query"]["term"]["id"]["value"]=id
    QUERY_BY_ID["query"]["match"]["id"]=id
    response = requests.post(SEARCH_URL, data=json.dumps(QUERY_BY_ID), headers=headers)
    return response.json()
