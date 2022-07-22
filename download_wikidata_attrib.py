import pandas as pd
from pathlib import Path
import requests


def extract_image_license(image_name: str):
    """
    get licence for image stored on wikimedia commons
    """

    start_of_end_point_str = (
        "https://commons.wikimedia.org" "/w/api.php?action=query&titles=File:"
    )
    end_of_end_point_str = (
        "&prop=imageinfo&iiprop=user"
        "|userid|canonicaltitle|url|extmetadata&format=json"
    )
    result = requests.get(start_of_end_point_str + image_name + end_of_end_point_str)
    result = result.json()
    page_id = next(iter(result["query"]["pages"]))
    try:
        image_info = result["query"]["pages"][page_id]["imageinfo"]
        licence = image_info[0]["extmetadata"]["UsageTerms"]["value"]
    except:
        licence = ""
    return licence


def extract_wiki_license(image_name: str):

    start_of_end_point_str = (
        "https://en.wikipedia.org" "/w/api.php?action=query&titles=File:"
    )
    end_of_end_point_str = (
        "&prop=imageinfo&iiprop=user"
        "|userid|canonicaltitle|url|extmetadata&format=json"
    )
    result = requests.get(start_of_end_point_str + image_name + end_of_end_point_str)
    result = result.json()
    page_id = next(iter(result["query"]["pages"]))
    image_info = result["query"]["pages"][page_id]["imageinfo"]
    metadata = image_info[0]["extmetadata"]
    if "UsageTerms" in metadata:
        return metadata["UsageTerms"]["value"]
    elif "Permission" in metadata:
        return metadata["Permission"]["value"]


def get_licence(image_name: str):
    value = ""
    source = ""
    try:
        value = extract_image_license(image_name)
        if value:
            source = "https://commons.wikimedia.org/wiki/File:" + image_name
        else:
            value = extract_wiki_license(image_name)
            if value:
                source = "https://en.wikipedia.org/wiki/File:" + image_name
    except Exception:
        pass
    return value, source


def process_sources():
    """
    go through wikipedia sourced images and get their attribution info
    """
    df = pd.read_csv(Path("source", "wikidata_sources.csv"))
    df = df.loc[df["do_not_use"] == 0]
    df["commons_files"] = df["url"].str.split("/").str[-1]
    print(f"Processing {len(df)} images")
    results = df["commons_files"].apply(get_licence)
    df["licence"] = results.str[0]
    df["urls"] = results.str[1]

    df.to_csv(Path("source", "attrib", "wikidata_sources_with_attrib.csv"), index=False)


if __name__ == "__main__":
    process_sources()
