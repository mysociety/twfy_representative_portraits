#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Downloads thumbnails
of MPs from parliament API and wikipedia.

Does four passes:
1. Offical API
2. Wikipedia images where wikidata has a twfy_id
3. Wikipedia images where joined on name
4. Wikidata images where there is a twfy_id

4 overlaps with 2, but with older images so deprioritised. 

Wikipedia images consult an image list to make sure they haven't
previously been downloaded and discarded.

"""

import csv
import json
import re
import shutil
import sys
import time
from xml.etree import ElementTree
from collections import Counter
from functools import lru_cache
from pathlib import Path
from tempfile import gettempdir
from typing import Optional, Tuple
from urllib.parse import unquote
from urllib.request import urlopen, urlretrieve

import pandas as pd
import requests
import wikipedia
from PIL import Image, ExifTags
from popolo_data.importer import Popolo

small_image_folder = Path("web", "mps")
large_image_folder = Path("web", "mpsL")
uk_parl_image_folder = Path("source", "mpsOfficial")
welsh_parl_image_folder = Path("source", "welshOfficial")
wikidata_image_folder = Path("source", "mpsWikidata")
manual_image_folder = Path("source", "manual")

# query to get images stored in wikidata (should be used secondary to the wikipedia sources)
wikidata_query = """
SELECT DISTINCT ?person ?personLabel ?partyLabel ?twfy_id ?image {
 ?person p:P39 ?positionStatement .
 ?positionStatement ps:P39 [wdt:P279* wd:Q16707842] .  # all people who held an MP position
 ?person wdt:P18 ?image .
 ?person wdt:P2171 ?twfy_id . 
  
 SERVICE wikibase:label { bd:serviceParam wikibase:language 'en' }
}
ORDER BY ?start
"""

# query to make twfy_ids to wikipedia page via wikidata
wikidata_to_wikipedia_query = """
prefix schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT DISTINCT ?person ?twfy_id ?article WHERE {
     ?person p:P39 ?positionStatement .
     ?positionStatement ps:P39 [wdt:P279* wd:Q16707842] .
     ?person wdt:P2171 ?twfy_id . # with twfy id
    OPTIONAL {
      ?article schema:about ?person .
      ?article schema:inLanguage "en" .
      FILTER (SUBSTR(str(?article), 1, 25) = "https://en.wikipedia.org/")
    }
} 
"""

# query to make twfy_ids to wikipedia page via wikidata names
unided_wikipedia_query = """
prefix schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT DISTINCT ?person ?personLabel ?givenLabel ?familyLabel ?twfy_id ?article WHERE {
     ?person p:P39 ?positionStatement .
     ?positionStatement ps:P39 [wdt:P279* wd:Q16707842] .
     OPTIONAL { ?person wdt:P2171 ?twfy_id . } # with twfy id
     OPTIONAL { ?person wdt:P735 ?given . }
     OPTIONAL { ?person wdt:P734 ?family . }
    OPTIONAL {
      ?article schema:about ?person .
      ?article schema:inLanguage "en" .
      FILTER (SUBSTR(str(?article), 1, 25) = "https://en.wikipedia.org/")
    }
 SERVICE wikibase:label { bd:serviceParam wikibase:language 'en' }
} 
"""

WIKI_REQUEST = "http://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&piprop=original&titles="


def download_manual_attrib():
    """
    Download google sheet with attribution info for new images
    """
    url = "https://docs.google.com/spreadsheet/ccc?key=11QYlZEh3xGsRjw-2i3ackqdV_ECki80-Xae6MAN-eJs&output=csv"
    df = pd.read_csv(url)
    df.to_csv(Path("source", "attrib", "manual_attrib.csv"), index=False)


def combine_attribs():
    """
    combine all attribution files into one json
    """
    # prefer legacy, wikipedia, official

    legacy = pd.read_csv(Path("source", "attrib", "legacy_attrib.csv"))
    legacy["source"] = "legacy"

    # get wiki data sources
    wiki = pd.read_csv(Path("source", "attrib", "wikidata_sources_with_attrib.csv"))
    wiki = wiki[["id", "licence", "urls"]]
    wiki = wiki.rename(
        columns={
            "id": "person_id",
            "licence": "photo_attribution_text",
            "urls": "photo_attribution_link",
        }
    )
    wiki["source"] = "wikipedia"

    # get parliament sources
    parl = pd.read_csv(Path("source", "attrib", "parliament_attrib.csv"))
    parl["source"] = "parliament"

    # get Senedd sources
    parl = pd.read_csv(Path("source", "attrib", "senedd_attrib.csv"))
    parl["source"] = "senedd"

    # get manual sources
    man = pd.read_csv(Path("source", "attrib", "manual_attrib.csv"))
    man["source"] = "manual"

    # combine and convert to long data format
    df = (  # type: ignore
        pd.concat([parl, wiki, legacy, man])
        .drop(columns=["source"])
        .melt("person_id", var_name="data_key", value_name="data_value")
        .loc[lambda df: ~df["data_value"].isna()]
    )

    df.to_json(Path("web", "attribution.json"), orient="records")


def clean_name(name: str):
    return re.sub("[^A-Za-z0-9]+", "", name).lower()


@lru_cache
def get_banned_wikidata():
    """
    get list of wikidata stuff already screened out
    """
    df = pd.read_csv(Path("source", "wikidata_sources.csv"))
    banned_urls = list(df["url"][df["do_not_use"] == 1])
    return banned_urls


def get_image_from_wikipedia(page: str):
    """
    given a wikipedia page, get the main image
    """
    unescaped = unquote(page)
    wkpage = wikipedia.WikipediaPage(title=unescaped)
    title = wkpage.title
    response = requests.get(WIKI_REQUEST + title)
    json_data = json.loads(response.text)
    print("getting image url from wikipedia")
    try:
        img_link = list(json_data["query"]["pages"].values())[0]["original"]["source"]
        return img_link
    except KeyError:
        return None


def get_wikipedia():
    """
    get wikipedia images based in twfy_ids in wikidata
    """
    url = "https://query.wikidata.org/sparql"

    r = requests.get(
        url, params={"format": "json", "query": wikidata_to_wikipedia_query}
    )
    data = r.json()
    for person in data["results"]["bindings"]:
        twfy_id = person["twfy_id"]["value"]
        if "article" in person:
            wikipedia_url = person["article"]["value"]
            wikipedia_title = wikipedia_url.split("/")[-1]
            print(wikipedia_title)
            image_url = get_image_from_wikipedia(wikipedia_title)
            if image_url:
                get_wiki_image(image_url, twfy_id)


def get_idless_wikipedia(override: bool = False):
    """
    get images where there is a direct name but not id match in wikipedia
    """
    url = "https://query.wikidata.org/sparql"
    twfy_name_to_id = get_name_to_id_lookup()
    print("getting query")
    r = requests.get(url, params={"format": "json", "query": unided_wikipedia_query})
    data = r.json()
    print("fetched query")
    for person in data["results"]["bindings"]:
        twfy_id = person.get("twfy_id", {"value": None})["value"]
        if twfy_id:
            continue
        full_label = person["personLabel"]["value"]
        given = person.get("givenLabel", {"value": ""})["value"]
        family = person.get("familyLabel", {"value": ""})["value"]
        joined = given + " " + family
        full_label = clean_name(full_label)
        alt_label = clean_name(joined)
        twfy_id = twfy_name_to_id.get(full_label, twfy_name_to_id.get(alt_label, None))
        if twfy_id and "article" in person:
            twfy_id = twfy_id.split("/")[-1]
            print(twfy_id, person["article"]["value"])
            wikipedia_url = person["article"]["value"]
            wikipedia_title = wikipedia_url.split("/")[-1]
            print(wikipedia_title)
            dest_path = wikidata_image_folder / "{0}.jpg".format(twfy_id)
            if override == False and dest_path.exists():
                print("downloaded, skipping")
                continue
            image_url = get_image_from_wikipedia(wikipedia_title)
            if image_url:
                get_wiki_image(image_url, twfy_id)


def get_wikidata():
    """
    download images stored in wikidata by twfy_id
    - lower priority as less well updated than wikipedia
    """
    url = "https://query.wikidata.org/sparql"

    r = requests.get(url, params={"format": "json", "query": wikidata_query})
    data = r.json()
    for person in data["results"]["bindings"]:
        twfy_id = person["twfy_id"]["value"]
        image_url = person["image"]["value"]
        get_wiki_image(image_url, twfy_id)


def get_wiki_image(image_url: str, twfy_id: int, override: Optional[bool] = False):
    """
    given an image on wikipedia, download
    """
    banned = get_banned_wikidata()
    if image_url in banned:
        return None
    ext = image_url.split(".")[-1].lower()
    dest_ext = ext
    if dest_ext.lower() in ["gif"]:
        dest_ext = "jpg"
    if ext in ["svg", "pdf", "tif"]:  # coat of arms or something
        return None
    filename = "{0}.{1}".format(twfy_id, ext)
    uk_parl_filename = uk_parl_image_folder / "{0}.jpg".format(twfy_id)
    if uk_parl_filename.exists():
        return None
    temp_path = Path(gettempdir(), filename)
    dest_path = wikidata_image_folder / filename
    if override == False and dest_path.exists():
        return None
    urlretrieve(image_url, temp_path)
    print("downloaded: {0}".format(image_url))
    try:
        image = Image.open(temp_path)
    except Exception:
        return None
    image.thumbnail((260, 346), resample=Image.ANTIALIAS)
    image.save(dest_path, quality=95)
    image.close()
    temp_path.unlink()
    store_wiki_source(image_url, twfy_id)


def store_wiki_source(image_url, twfy_id):
    """
    add the source for this image to the cache
    """
    with open(Path("source", "wikidata_sources.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([twfy_id, image_url, 0])


def get_name_to_id_lookup():
    """
    create id lookup from popolo file
    where someone's name is unique
    try and map to twfy id
    """
    people_url = "https://github.com/mysociety/parlparse/raw/master/members/people.json"
    pop = Popolo.from_url(people_url)
    count = 0
    lookup = {}
    print("Creating name to id lookup")
    all_names = []

    def add_name(reduced_name, id):
        all_names.append(reduced_name)
        lookup[reduced_name] = id

    for p in pop.persons:
        id = p.id
        for name in p.other_names:
            possible_names = []
            if "given_name" in name and "family_name" in name:
                possible_names.append(name["given_name"] + " " + name["family_name"])
            if "additional_name" in name:
                possible_names.append(name["additional_name"])

            # only add one copy of each reduced name
            possible_names = [clean_name(x).lower() for x in possible_names]
            possible_names = list(set(possible_names))
            for p in possible_names:
                add_name(p, id)

    # if the same name leads to multiple ids, delete
    c = Counter(all_names)
    for k, v in c.items():
        if v > 1:
            del lookup[k]

    return lookup


def get_id_lookup(scheme: str) -> dict:
    """
    create id lookup from popolo file
    convert scheme ID to parlparse
    """
    people_url = "https://github.com/mysociety/parlparse/raw/master/members/people.json"
    pop = Popolo.from_url(people_url)
    count = 0
    lookup = {}
    print("Creating id lookup")
    for p in pop.persons:
        id = p.id
        identifier = p.identifier_value(scheme)
        if identifier:
            lookup[identifier] = id[-5:]
            count += 1
    print(count, len(pop.persons))
    return lookup


image_format = (
    "https://members-api.parliament.uk/api/Members/{0}/Portrait?CropType=ThreeFour"
)


def download_and_resize(
    mp_id: int, parlparse: str, override: Optional[bool] = False
) -> str | None:
    """
    download and retrieve the three-four sized
    offical portrait
    """
    filename = f"{parlparse}.jpg"
    vlarge_path = uk_parl_image_folder / filename
    temp_path = Path(gettempdir(), f"{mp_id}.jpg")
    image_url = image_format.format(mp_id)
    api_url = f"https://members-api.parliament.uk/api/Members/{mp_id}"
    attempts = 0
    api_results = None
    while attempts <= 5 and api_results is None:
        try:
            api_results = json.loads(urlopen(api_url).read())
        except Exception as e:
            print("API fetch error, sleeping and retrying")
            attempts += 1
            time.sleep(5)
    if api_results is None:
        raise ValueError("API Fetch Error")
    thumbnail_url = api_results["value"].get("thumbnailUrl", "")
    if "members-api" not in thumbnail_url:
        print("no offical portrait")
        return None
    try:
        urlretrieve(image_url, temp_path)
    except Exception:
        return None
    print(f"downloaded: {image_url}")
    try:
        image = Image.open(temp_path)
    except Exception:
        return None
    image.save(vlarge_path, quality=100)
    image.close()
    temp_path.unlink()
    return image_url


def get_uk_parl_images(override: bool = False):
    """
    fetch image if available from offical source
    """
    lookup = get_id_lookup("datadotparl_id")

    urls = []
    ids = []

    for datadotparl, parlparse in lookup.items():
        print(datadotparl, parlparse)
        filename = f"{parlparse}.jpg"
        uk_parl_path = uk_parl_image_folder / filename
        if uk_parl_path.exists() is False or override:
            url = download_and_resize(datadotparl, parlparse, override)
            if url:
                urls.append(url)
                ids.append(parlparse)

    df = pd.DataFrame({"person_id": ids, "photo_attribution_link": urls})
    df["photo_attribution_text"] = "© Parliament (CC-BY 3.0)"
    df.to_csv(Path("source", "attrib", "parliament_attrib.csv"), index=False)


def get_welsh_parl_images(override: bool = False):
    """
    fetch image if available from offical source
    """
    lookup = get_id_lookup("senedd")

    urls = []
    ids = []

    api_url = 'https://business.senedd.wales/mgwebservice.asmx/GetCouncillorsByWard'
    api_results = ElementTree.parse(urlopen(api_url))
    for item in api_results.findall('.//councillor'):
        id = item.find('councillorid').text
        parlparse = lookup[id]
        print(id, parlparse)
        image_url = item.find('photobigurl').text
        filename = f"{parlparse}.jpeg"
        welsh_parl_path = welsh_parl_image_folder / filename
        if welsh_parl_path.exists() is False or override:
            urlretrieve(image_url, welsh_parl_path)
            urls.append(image_url)
            ids.append(parlparse)

    df = pd.DataFrame({"person_id": ids, "photo_attribution_link": urls})
    df["photo_attribution_text"] = "© Senedd (CC-BY 4.0)"
    df.to_csv(Path("source", "attrib", "senedd_attrib.csv"), index=False)


def ids_from_directory(dir: Path) -> set:
    """
    get twfy ids used in images in this dir
    """
    ids = [x.stem for x in dir.iterdir()]
    return set(ids)


def overlap_report():
    """
    quick summary on where we have large images and where
    there are still small images
    """

    big = ids_from_directory(uk_parl_image_folder)
    wikidata = ids_from_directory(wikidata_image_folder)
    big.update(wikidata)
    small = ids_from_directory(small_image_folder)
    missing = small.difference(big)
    missing_small = big.difference(small)
    print(f"There are big images for {len(big)}")
    print(f"There are small images for {len(small)}")
    print(f"Small but not big: {len(missing)}")
    print(f"Big but not small: {len(missing_small)}")


def pad_to_size(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """
    resize images and add padding to get to right size.
    """
    final = Image.new(image.mode, size, "white")
    thumbnail = image.copy()
    thumbnail.thumbnail(size, resample=Image.ANTIALIAS)
    final.paste(thumbnail)
    return final


def downsize(image_path: Path, only_small: Optional[bool] = False):
    """
    downsize a particular image to the small and big
    folders
    """
    print(image_path)
    filename = image_path.name
    small_path = small_image_folder / filename
    large_path = large_image_folder / filename
    image = Image.open(image_path)

    # respect EXIF tags on rotation
    reverse_tags = dict((v, k) for k, v in ExifTags.TAGS.items())
    orientation = reverse_tags["Orientation"]

    exif = image.getexif()
    if orientation in exif:
        if exif[orientation] == 3:
            print("Rotating 180")
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            print("Rotating 270")
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            print("Rotating 90")
            image = image.rotate(90, expand=True)

    if only_small is False:
        large_image = pad_to_size(image, (120, 160))
        large_image.save(large_path, quality=95)
    small_image = pad_to_size(image, (60, 80))
    small_image.save(small_path, quality=95)
    image.close()


def make_large_from_folder(folder: Path):
    """
    for each file in folder, create files that match
    small and big size
    """
    for f in folder.iterdir():
        downsize(f)


def prefer_jpg(folder: Path):
    """
    If a directory has "jpg" and "jpeg"
    remove "jpeg"
    """
    ids = ids_from_directory(folder)
    for i in list(ids):
        jpeg = folder / "{0}.jpeg".format(i)
        jpg = folder / "{0}.jpg".format(i)
        if jpg.exists() and jpeg.exists():
            jpeg.unlink()


def prefer_lower_case(folder: Path):
    """
    Rename image suffixes to lowercase
    """
    formats = ["PNG", "JPEG", "JPG"]
    for file_format in formats:
        for f in folder.glob(f"*.{file_format}"):
            print(f"renaming {f.name} with uppercase extention")
            f.rename(f.with_suffix(f".{file_format.lower()}"))


def copy_legacy():
    """
    Copy files present before this system as a base
    """
    legacy = Path("source", "legacy")
    web = Path("web")

    for f in ["mps", "mpsL"]:
        source = legacy / f
        dest = web / f
        shutil.copytree(source, dest, dirs_exist_ok=True)


def prepare_images(manual_only: Optional[bool] = False):
    """
    Update portrait folders and attrib info
    from source folders
    """
    download_manual_attrib()

    # rename to lower case
    for f in [
        uk_parl_image_folder,
        welsh_parl_image_folder,
        wikidata_image_folder,
        manual_image_folder,
        large_image_folder,
        small_image_folder,
    ]:
        prefer_lower_case(f)

    if manual_only is False:
        copy_legacy()
        make_large_from_folder(wikidata_image_folder)
        make_large_from_folder(welsh_parl_image_folder)
        make_large_from_folder(uk_parl_image_folder)
    make_large_from_folder(manual_image_folder)
    prefer_jpg(large_image_folder)
    prefer_jpg(small_image_folder)
    combine_attribs()


if __name__ == "__main__":

    args = sys.argv[1:]
    if len(args) == 0:
        args = ["all"]

    def arg_test(*potential):
        for p in ["all"] + list(potential):
            if p in args:
                return True
        return False

    if arg_test("fetch_all", "fetch_uk_parl"):
        get_uk_parl_images()
    if arg_test("fetch_all", "fetch_welsh_parl"):
        get_welsh_parl_images()
    if arg_test("fetch_official_all"):
        get_uk_parl_images(override=True)
        get_welsh_parl_images(override=True)
    if arg_test("fetch_all", "fetch_wiki"):
        get_wikipedia()
        get_idless_wikipedia()
        get_wikidata()
    if arg_test("prepare"):
        prepare_images()
    if arg_test("manual"):
        prepare_images(manual_only=True)
    if arg_test("report"):
        overlap_report()
