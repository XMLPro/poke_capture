import os
import glob
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
from time import sleep
from random import random

url = "https://www.serebii.net{}".format
for filename in tqdm(glob.glob("../serebii_detail/*")):
    with open(filename, "rb") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for img in tqdm(soup.select(".dextable img[src*='icon']")):
        target = url(img["src"])
        sleep(random() * 3 + 0.3)
        with open(target.split("/")[-1], "wb") as f:
            response = requests.get(target)
            f.write(response.content)
