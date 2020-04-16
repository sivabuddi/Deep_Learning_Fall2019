import urllib.request
from inscriptis import get_text

count =1
with open("links.txt","r") as list_links:
    for url in list_links:
        html = urllib.request.urlopen(url).read().decode('utf-8')
        text = get_text(html)
        print(text)
        print("URL completed:",count)
        count +=1