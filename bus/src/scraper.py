
import requests
from bs4 import BeautifulSoup
import os
import urllib.request
import logging

class ImageScraper:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def fetch_image_urls(self, search_term):
        url = f"https://www.google.com/search?hl=en&tbm=isch&q={search_term}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        images = [img['src'] for img in soup.find_all('img') if 'src' in img.attrs]
        return images


    def save_images(self, images, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        

        for img_url in images:
            img_name = os.path.join(directory, os.path.basename(img_url))
            urllib.request.urlretrieve(img_url, img_name)