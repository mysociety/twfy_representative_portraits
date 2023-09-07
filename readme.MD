# Representatives portraits

This is a repository to hold the portrait images used for representatives in TheyWorkForYou.

`python process.py` will run through the process of checking the official API and wikipedia/wikidata looking for new images. Part of this can be run in turn with additional arguments:

* `fetch_official_all` - download all official portraits and match to TWFY IDs.
    * `fetch_uk_parl` - download missing official UK Parliament portraits and match to TWFY IDs.
    * `fetch_welsh_parl` - download missing official Welsh Parliament portraits and match to TWFY IDs.
* `fetch_wiki` - fetch images from wikipedia based on TWFY ID, wikidata ID, or name (in some cases).
* `prepare` - downscale the fetched images to the two sizes expected by TWFY.
* `manual` - will only downscale images in the manual source directory, and pull down the attrib spreadsheet. 
* `report` - quick summary of how many images in only one size are avaliable. 

The two sizes of images used by TheyWorkForYou are stored under `web/mps` and `web/mpsL`.

Slightly larger source images are collected at `source/mpsOfficial` and `source/mpsWikidata`.

Wikidata image urls/IDs are checked against a 'not allowed' list (`source/wikidata_sources.csv`) that were manually screened previously to remove image that:

* Are not representations of that MP.
* Are caricatures rather than portraits. 
* Not a good 'portrait' image (too small in frame).
* Avaliable through the official API.

There will be errors in both directions and this is meant as a way of making updates from Wikipedia more repeatable without reimporting images previously screened.

Images that are not sourced from these locations but that were in TheyWorkForYou are included in a legacy folder.

The priority order of what is included (from highest to lowest): Manual, Parliament, Wikipedia, Legacy. 

To update attribution, run `download_wikidata_attrib.py` and then tidy up `source\attrib\wikidata_sources_with_attrib.csv`. Run the `combine_attributes` notebook to review how this is merged into `attribution.json`.

To add attribution details for manually added images, modify https://docs.google.com/spreadsheets/d/11QYlZEh3xGsRjw-2i3ackqdV_ECki80-Xae6MAN-eJs/edit#gid=0

## Uploading a manual image through GitHub

Images in the `manual` folder take priority over other sources. You can add images to this folder through GitHub. 

* Name the file on your computer '{mp_person_number}.jpg' or similar (e.g. 13956.jpg)
* Go to https://github.com/mysociety/twfy_representative_portraits/tree/main/source/manual
* CLick Add file > Upload files
* Drag and drop file or select your file.
* Make a note in the commit box of what you are adding/changing and why.
* A github action will then run to resize the image.  After a few minutes, you should see a new commit saying 'Resized manual image'.  Clicking this commit [(example)](https://github.com/mysociety/twfy_representative_portraits/commit/2c2609971cab584a8740c0fcb03f502d4f6bc938) will show the change that has been made. 
* At this point, next time a deveoloper redeploys TheyWorkForYou - the new image will be loaded. 
