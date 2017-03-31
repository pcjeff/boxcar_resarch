import re
from  bs4  import  BeautifulSoup
import urllib2
import os

def get_urls_from_html(html):
    html = urllib2.unquote(html)
    soup = BeautifulSoup(html, 'lxml')
    urls = []
    for el in soup.find_all('a', class_='rg_l'):
        href = urllib2.unquote(el['href'])
        _u = re.findall('imgurl=(.*)\&imgref', href)
        if len(_u) > 0:
            u = _u[0]

        urls.append(u)
    return urls

def save(urls, outfile):
    with open(outfile, 'w+') as f:
        for u in urls:
            print>>f, u

urls_list = []
#html_files = [f for f in os.listdir('./boxcar_html/') if os.path.isfile(os.path.join('./boxcar_html/', f))]
html_files = ['skodaoctaviasedan.html']

for c in html_files:
    print c
    with open(os.path.join('/tmp3/pcjeff/boxcar/boxcar_html/', c)) as f:
        urls = get_urls_from_html(''.join(f.readlines()))
    print len(urls)
    urls_list.append(urls)
    
    outfile = '%s_imageurls.txt' % c
    save(urls, outfile)
