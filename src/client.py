
from flask import Flask
from flask import request
import os
from dateutil.parser import parse
import json

from paths import OUTPUT_DIR

app = Flask(__name__)


@app.route("/")
def hello_world():
    ror = request.args.get('ror')
    since_date_param = request.args.get("since")
    since_date = parse(since_date_param)
    print(f"Searching for {ror} since {since_date}")
    response = []
    for filename in os.listdir(OUTPUT_DIR):
        print(filename)
        if os.path.isdir(os.path.join(OUTPUT_DIR, filename)):
            if is_in_interval(filename, since_date):
                with open(os.path.join(OUTPUT_DIR, filename, "affiliation.json"), "r") as affiliations:
                    response.append(process_affiliations(affiliations, ror))
        else:
            continue
    print(f"returning {len(response)} results")
    return json.dumps(response)


def is_in_interval(filename, since_date):
    try:
        filedate = parse(filename.split("_")[0])
        return filedate > since_date
    except:
        return False


def process_affiliations(affiliations_file, ror):
    affiliations = json.load(affiliations_file)
    results = []
    for aff in affiliations:
        for author in aff["affiliation_info"]:
            if isinstance(author["affiliation"], (list, tuple)):
                if author["affiliation"][0]["ror_id"]:
                    aff_ror_id = author["affiliation"][0]["ror_id"][0]
                    if ror == aff_ror_id:
                        doi = aff["DOI"]
                        print(f"Found {doi}")
                        results.append(aff)
                        continue
            else:
                if author["affiliation"]["ror_id"]:
                    aff_ror_id = author["affiliation"]["ror_id"][0]
                    if ror == aff_ror_id:
                        doi = aff["DOI"]
                        print(f"Found {doi}")
                        results.append(aff)
                        continue
    print(f"Processed {len(affiliations)} DOIs returning {len(results)} matches")
    return results


if __name__ == "__main__":
    if os.getenv("debug"):
        app.run(debug=True, port=5007)
    else:
        from waitress import serve
        serve(app, host="0.0.0.0", port=5007)
