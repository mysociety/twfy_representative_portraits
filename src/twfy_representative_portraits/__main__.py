from .process import (
    get_uk_parl_images,
    get_welsh_parl_images,
    get_wikipedia,
    get_idless_wikipedia,
    get_wikidata,
    prepare_images,
    overlap_report,
)
import typer

app = typer.Typer()


@app.command()
def fetch_all():
    get_uk_parl_images()
    get_welsh_parl_images()
    get_wikipedia()
    get_idless_wikipedia()
    get_wikidata()


@app.command()
def fetch_wiki():
    get_wikipedia()
    get_idless_wikipedia()
    get_wikidata()


@app.command()
def fetch_uk_parl():
    get_uk_parl_images()


@app.command()
def fetch_welsh_parl():
    get_welsh_parl_images()


@app.command()
def fetch_official_all():
    get_uk_parl_images(override=True)
    get_welsh_parl_images(override=True)


@app.command()
def prepare():
    prepare_images()


@app.command()
def prepare_manual():
    prepare_images(manual_only=True)


@app.command()
def report():
    overlap_report()


if __name__ == "__main__":
    app()
