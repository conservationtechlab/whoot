"""Xeno-Canto Data Metadata Downloader and Search Module."""
import os
import urllib.parse
import json
import requests


class XenoCantoDownloader():
    """Handler for Xeno-Canto API.

    Note: Requires an API key from env var "XC_API_KEY".
    Third version of the Xeno-Canto API is used here.
    """
    def __init__(self, api_key=None):
        """Creates the Xeno-Canto Downloader.

        Args:
            api_key (str): API key for Xeno-Canto API.
                If None, looks for env var "XC_API_KEY"
        """
        self.endpoint_url = "https://xeno-canto.org/api/3/recordings"
        self.api_key = os.environ["XC_API_KEY"] if api_key is None else api_key
        assert self.api_key is not None, \
            "API KEY MISSING: Put API key in Environment Var!"

    def __call__(self,
                 query=None,
                 loc=None,
                 ):
        r"""Download XC data.

        Initally, this was intended to be used to build queries
        So more args were planned (hence loc). In practice, it was easier
        to build queries by hand ¯\_(ツ)_/¯

        You can pull the query you want from the url on the website if you
        are manually searching for thigns there. Its the same syntax.

        Also is useful for debugging issues there

        Args:
            query (str/None): Search query string see XC Search Tags
            loc (str/None): Location string for search query
        """
        if query is None:
            query = self.build_query(
                loc=loc,
            )

        page_datas = []
        page_data = self.get_page(query, page=1)
        page_datas.append(page_data)

        # Get rest of data!
        for i in range(2, page_data["numPages"] + 1):
            page_data = self.get_page(query, page=i)
            page_datas.append(page_data)

        return page_datas

    def concat_recording_data(self, page_datas):
        """Concatinate recording data from multiple pages.

        Args:
            page_datas (list): list of page data dicts
        """
        new_page_data = []
        for page_data in page_datas:
            new_page_data = new_page_data + page_data["recordings"]
        return new_page_data

    def build_query(
        self,
        loc="San Diego, California, United States of America",
        # box=None,
    ):
        """Builds a query string for Xeno-Canto API.

        See https://xeno-canto.org/help/search
        Args:
            loc (str): Location string for search query
        """
        search_tags = ""
        if loc is not None:
            search_tags += f"loc:\"{loc}\"+"
        # Remove trailing +
        return search_tags[:-1]

    def get_page(self, query, page=1):
        """Get a page of results from Xeno-Canto API.

        Args:
            query (str): Search query string see XC Search Tags
            page (int): Page number to retrieve
        """
        res = requests.get(
            self.endpoint_url + "?" + urllib.parse.urlencode({
                "query": query,
                "key": self.api_key,
                "page": page
            }),
            timeout=100
        )
        if res.status_code == 200:
            return json.loads(res.text)

        return {}
    # def download_files(self, data):
    #     if type(data) == dict:
    #        data = self.concat_recording_data(self, data)
    #     for recording in data:
    #         requests


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description='Input Directory Path'
    #     )
    # parser.add_argument('meta', type=str,
    #                     help='Path to metadata csv')
    # args = parser.parse_args()
    xcd = XenoCantoDownloader()
    print(xcd())
