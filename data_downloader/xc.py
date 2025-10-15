import requests
import os
import json
import urllib.parse

class XenoCantoDownloader():
    def __init__(self, api_key=None):
        self.endpoint_url = "https://xeno-canto.org/api/3/recordings"
        self.api_key = os.environ["XC_API_KEY"] if api_key is None else api_key
        assert self.api_key is not None, "API KEY MISSING: Put API key in Enviroment Var!"

    def __call__(self,
                 query = None,
                 loc=None,
                 box=None):
        
        if query is None:
            query = self.build_query(
                loc=loc,
                box=None,
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
        new_page_data = []
        for page_data in page_datas:
            new_page_data = new_page_data + page_data["recordings"]
        return new_page_data

    def build_query(
        self,
        loc="San Diego, California, United States of America",
        box=None,
    ):
        search_tags = ""
        if loc is not None:
            search_tags += f"loc:\"{loc}\"+"
        return search_tags[:-1] #remove last +

    def get_page(self, query, page=1):
        res = requests.get(self.endpoint_url + "?"+ urllib.parse.urlencode({
            "query": query,
            "key": self.api_key,
            "page": page
        }))
        if res.status_code == 200:
            return json.loads(res.text)
        else:
            {}
    
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