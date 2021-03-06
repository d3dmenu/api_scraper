import os, sys # System
import re # Regular Expression
import csv # Spread Sheet
import json # Export data json files
import joblib, pickle # Load model
import random, string
import time # Performance time
import uvicorn # Server deploy api
import warnings # Disable message warnings
import numpy as np
import tldextract # Extract url name

''' Beautiful Soup '''
from bs4 import BeautifulSoup # Find elements class from site.
from bs4.element import Comment # Cleansing synctax.
from urllib.request import urlopen

from fastapi import FastAPI, Form # Library api server
# Machine Learning Random Forest Algorithm
from sklearn.ensemble import RandomForestClassifier

from selenium import webdriver # Chrome driver scraper .
from selenium.webdriver.chrome.options import Options # Configuration mode.
from selenium.webdriver.support.ui import WebDriverWait # Wait connect to website.
from selenium.webdriver.support import expected_conditions as EC # Check element from website.
from selenium.webdriver.common.by import By # Wait elementy by (class, id, css_selector etc).
from selenium.common.exceptions import TimeoutException # Request timout

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  
from pythainlp.corpus import thai_stopwords
from pythainlp import word_tokenize


''' Configuration Program '''

# Disable Warning
warnings.filterwarnings('ignore') 
# List Data test
Model_Test = []
# Dictionary data.
Class_Properties = {}
# Chrome setting (change mode headless and visible).
Chrome_options = True
# When program finish -> Export JSON.
Export_JSON = True
# Scaning qualify class
Qualify = True
# Loading target website success and wait 5 sec.
Delay = 2
# Data stop word (en / th).
data_stopwords = list(stopwords.words('english')) + list(thai_stopwords())
# Pattern Address
address = ['average', 'dept', 'size', 'weight', 'height', 'word', 'a', 'div', 'em', 'h1', 'h2', 'h3', 'h4', 'h5', 'label', 'li', 'p', 'section', 'span', 'strong', 'td', 'tr', 'ul']
stringContent = ""


# Create api with fastapi
app = FastAPI()

# Navigation
@app.get("/")
async def main():
    return 'Deploy Model NGate Engine'

# Navigation
@app.get("/api/v1/manual/")
async def api_manual(Z_test: list = Form(...)): 
    return "System Coming soon... "

@app.get("/api/v1/predict/")
async def api_predicted(url: str = Form(...)):
    if url == "": return "URL not invalid"
    ''' Optional Function Program '''
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(NpEncoder, self).default(obj)

    def xpath_soup(element):
        components = []
        child = element if element.name else element.parent
        for parent in child.parents:
            siblings = parent.find_all(child.name, recursive=False)
            components.append(
                child.name
                if siblings == [child] else
                '%s[%d]' % (child.name, 51 + siblings.index(child))
                )
            child = parent
        components.reverse()
        return '/%s' % '/'.join(components)

    def get_proc(clname, source):
        try:
            soup = BeautifulSoup(source, "lxml")
            tag = soup.find(class_=clname)
            elem = soup.find(tag.name, class_=clname)
            return [elem, tag.name]
        except:
            pass

    def export_json(data):

        try:
            app_json = json.dumps(data, cls=NpEncoder)
            return app_json
        except Exception as e:
            return e

    def progress(count, total, status=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()

    def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        total = len(iterable)
        # Progress Bar Printing Function
        def printProgressBar (iteration):
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Initial Call
        printProgressBar(0)
        # Update Progress Bar
        for i, item in enumerate(iterable):
            yield item
            printProgressBar(i + 1)
        # Print New Line on Complete
        # print((f"{Fore.LIGHTWHITE_EX}Download complete"))
        print("Download complete! ")

    def converter(data):
        ''' Convert Base px '''
        result, value = [], 0

        for i in range(len(data)):
            if "px" in data[i]:
                value = data[i].replace("px", "")
                result.append(value)
            elif "em" in data[i] or "rem" in data[i]:
                if "em" in data[i]: 
                    value = data[i].replace("em", "")
                if "rem" in data[i]: 
                    value = data[i].replace("rem", "")
                value = float(value) * 16
                result.append(value)
            else:
                result.append(data[i])
            
        # fix weight (Font weight)
        # inherit bolder lighter => null 

        block = ["inherit", "bolder", "lighter"]
        
        if data[1] == "normal" or data[1] == "initial":
            result[1] = 400
        elif data[1] == "bold":
            result[1] = 700
        elif data[1] in block:
            result[1] = None

        # fix normal (line height)
        if data[0] == "normal" or data[0] == "initial":
            pass
        if data[1] == "normal" or data[1] == "initial":
            pass
        if data[2] == "normal" or data[2] == "initial":
            result[2] = float(result[0]) * 1.25

        return result

    def tag_visible(element):
        # blacklist = ['[document]','noscript', 'header', 'html', 'meta', 'head', 'input', 'script']
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    def text_from_html(body):
        soup = BeautifulSoup(body, 'html.parser')
        texts = soup.findAll(text=True)
        visible_texts = filter(tag_visible, texts)  
        return u" ".join(t.strip() for t in visible_texts)

    def get_TextHTML(Script):
        stringHtml = text_from_html(Script)
        stringContent = detect_word(stringHtml) 
        stringContent = [i for i in stringContent if i not in data_stopwords ]
        return stringContent

    def detect_word(text):
        proc = word_tokenize(text, engine='newmm', keep_whitespace=False)
        return proc

    def get_htmlTag(clname, soup):
        try:
            tag = soup.find(class_=clname)
            elem = soup.find(tag.name, class_=clname)
            return [elem, tag.name]
        except:
            print("Get Tag Error!")

    def text_from_html(body):
        soup = BeautifulSoup(body, 'html.parser')
        texts = soup.findAll(text=True)
        visible_texts = filter(tag_visible, texts)  
        return u" ".join(t.strip() for t in visible_texts)

    def get_TextHTML(Script):
        stringHtml = text_from_html(Script)
        stringContent = detect_word(stringHtml) 
        stringContent = [i for i in stringContent if i not in data_stopwords ]
        return stringContent

    def detect_word(text):
        proc = word_tokenize(text, engine='newmm', keep_whitespace=False)
        return proc

    def get_ratio(text):
        global data_stopwords, stringContent

        # detect word in stringText
        text_to_word = detect_word(text)
        
        # cleansing same word in stringText
        clean_same_word = []
        preprocess = [clean_same_word.append(x) for x in text_to_word if x not in clean_same_word]

        # cleansing text stop word
        contentText = [i for i in clean_same_word if i not in data_stopwords]
        ratioRate = 0
        for idx in range(len(contentText)):
            word = contentText[idx]
            if stringContent.count(word):
                ratioRate += stringContent.count(word)
        return round((ratioRate / len(contentText)), 2), len(contentText)
    
    def d2l(tag):
        global address
        index = address.index(tag)
        return index

    def get_className(soup):
        # Global variable.
        global Class_Properties, Model_Test
        # Properties.
        # prop = ["font-size", "font-wieght", "line-height", "color", "font"]
        section, ratio, uid = None, 0, None

        # Loop get class name with bs4 (soup).
        for element in progressBar(soup.find_all(class_= True), prefix = 'Progress:', suffix = 'Complete', length = 50):
            # clname is Class name.
            clname = element["class"]
            # Check same class in site
            if len(clname) == 1 and clname[0] not in Class_Properties:
                #  elem is list same class (xpath)
                elem = soup.find_all(class_= clname[0])
                # _obj is list same class.
                _obj = driver.find_elements_by_class_name(clname[0])
                for idx in range(len(_obj)):
                    # Add class name to dict.
                    # Class already change name > ex "???#1".
                    if clname[0] not in Class_Properties:
                        # Create data on dictionary.
                        Class_Properties[clname[0]] = {}
                        section, uid = Class_Properties[clname[0]], clname[0]
                    else:
                        # Case class same in site.
                        multiple_name = "%s_#%s" % (clname[0], idx)
                        Class_Properties[multiple_name] = {}
                        section, uid = Class_Properties[multiple_name], multiple_name

                    ''' List CSS Properties '''
                    try:
                        content = _obj[idx].get_attribute("textContent").splitlines()[0]
                        if content == "":
                            content = _obj[idx].text
                        # average word.
                        ratio, count_word = get_ratio(content)
                        # Get absolute xpath with elem.
                        find_xpath = xpath_soup(elem[idx])
                    except:
                        del Class_Properties[uid]
                        continue 
                    
                    if content != "":
                        try:
                            # Get Attributes.
                            size = _obj[idx].value_of_css_property('font-size')
                            weight = _obj[idx].value_of_css_property('font-weight')
                            height = _obj[idx].value_of_css_property('line-height')
                            # color = _obj[idx].value_of_css_property('color')
                            # font = _obj[idx].value_of_css_property('font')

                            # Preprocess format unit.
                            pre = converter([size, weight, height])
                            section["Font-size"] = float(pre[0])
                            section["Font-weight"] = int(pre[1])
                            section["Line-height"] = float(pre[2])
                            section["Word-content"] = count_word                        
                            # section["Color"] = color
                            # section["Font"] = font
                            section["Tag"] = _obj[idx].tag_name
                            section["Ratio"] = float(ratio)
                            section["Dept"] = find_xpath.count("/")
                            # section["Path"] = find_xpath
                            section["Text"] = content

                            ''' Save Data information to list Model Test '''
                            Load_Data = [
                                    section["Ratio"], section["Dept"], section["Font-size"],
                                    section["Font-weight"], section["Line-height"], section["Word-content"],
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            ]
                            Load_Data[d2l(section["Tag"])] = 1
                            Model_Test.append(Load_Data)
                        except:
                            del Class_Properties[uid]
                            continue
            
            elif len(clname) > 1 and ("." + ".".join(map(str, clname))) not in Class_Properties:
                multi = "." + ".".join(map(str, clname))
                #  elem is list same class (xpath).
                elem = soup.find_all(class_= " ".join(map(str, clname)))
                # _obj is list same class.
                try:
                    _obj = driver.find_elements_by_css_selector(multi)
                except:
                    continue

                for idc in range(len(_obj)):
                    # Add class name to dict.
                    # Class already change name > ex "???#1".
                    if multi not in Class_Properties:
                        # Create data on dictionary
                        Class_Properties[multi] = {}
                        section, uid = Class_Properties[multi], multi
                    else:
                        # Case class same in site
                        multiple_class = "%s_#%s" % (multi, idc)
                        Class_Properties[multiple_class] = {}
                        section, uid = Class_Properties[multiple_class], multiple_class

                    ''' List CSS Properties '''
                    try:
                        content = _obj[idc].get_attribute("textContent").splitlines()[0]
                        if content == "":
                            content = _obj[idc].text
                        ratio, count_word = get_ratio(content)
                        # Get absolute xpath with elem
                        find_xpath = xpath_soup(elem[idc])
                    except:
                        del Class_Properties[uid]
                        continue
        
                    if content != "":
                        try:
                            # Get Attributes
                            size = _obj[idc].value_of_css_property('font-size')
                            weight = _obj[idc].value_of_css_property('font-weight')
                            height = _obj[idc].value_of_css_property('line-height')
                            # color = _obj[idc].value_of_css_property('color')
                            # font = _obj[idc].value_of_css_property('font')

                            # Preprocess format unit 
                            pre = converter([size, weight, height])
                            section["Font-size"] = pre[0]
                            section["Font-weight"] = pre[1]
                            section["Line-height"] = pre[2]
                            section["Word-content"] = count_word
                            # section["Color"] = color
                            # section["Font"] = font
                            section["Tag"] = _obj[idc].tag_name
                            section["Ratio"] = ratio
                            section["Dept"] = find_xpath.count("/")
                            # section["Path"] = find_xpath
                            section["Text"] = content

                            ''' Save Data information to list Model Test '''
                            Load_Data = [
                                    section["Ratio"], section["Dept"], section["Font-size"],
                                    section["Font-weight"], section["Line-height"], section["Word-content"],
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                            ]
                            Load_Data[d2l(section["Tag"])] = 1
                            Model_Test.append(Load_Data)
                        except:
                            del Class_Properties[uid]
                            continue
                            
    ''' Chrome Driver Options Settings '''

    options = Options()
    options.add_argument("--incognito")
    options.add_argument("--headless")
    options.add_argument("--log-level=3")
    options.add_argument("--ignore-certificate-errors")
    
    # When Chrome configuration is TRUE.
    if Chrome_options:
        # Start Driver and add options to chrome.
        driver = webdriver.Chrome("chromedriver.exe", chrome_options = options)
    else:
        # Chrome option is False.
        driver = webdriver.Chrome("chromedriver.exe")

    driver.get(url)

    # Extract domain name
    extracted_domain = tldextract.extract(url)
    domainName = "{}".format(extracted_domain.domain)

    ''' Scroll Down Site '''
    SCROLL_PAUSE_TIME = 0.5
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    
    # Waiting Loading Website with WebDriverWait | Timesleep '''
    time.sleep(Delay)

    script = (driver.page_source).encode('utf-8', 'ignore')
    t1 = time.time() # Performance times

    stringContent = get_TextHTML(script)
    soup = BeautifulSoup(script, "html.parser")

    ''' Call Function Get Class on HTML Script '''
    get_className(soup)
    print(len(Class_Properties), len(Model_Test))

    # Performance time
    t2 = time.time() - t1
    print(f"Executed in {t2:0.2f} seconds.")

    # Predict data
    loaded_model = joblib.load("rforest.h7")
    predicted = loaded_model.predict(Model_Test)
    predict_proba = loaded_model.predict_proba(Model_Test)

    ''' Insert value after predict by decistion to dictionary (Infomation) '''

    num = 0

    for key, value in Class_Properties.items():
        if predicted[num] == 0:
            Class_Properties[key]["Class"] = "Title"
        elif predicted[num] == 1:
            Class_Properties[key]["Class"] = "Description"
        elif predicted[num] == 2:
            Class_Properties[key]["Class"] = "Price"
        else:
            Class_Properties[key]["Class"] = "Error"

        Class_Properties[key]["Proba"] = max(predict_proba[num])
        num += 1
    
    ''' Scan Class Qualify [Title Only] '''

    if Qualify:
        ''' Qualify with Beautiful Soup '''
        tag = soup.find_all('h1')
        for i in tag:
            name = "".join([random.choice(string.digits) for i in range(4)])
            section = Class_Properties[name] = {}
            section["Text"] = i.text
            section["Class"] = "Title"
        # Turn off chrome driver
    driver.close()
    # Export Dictionary to Json
    # export_json(Class_Properties)
    return Class_Properties


if __name__ == "__main__":
    uvicorn.run("main:app", host = "127.0.0.1", port = 8080)