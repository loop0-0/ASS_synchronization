import requests
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
#-insiall i + URL
i= 153
maw=["https://kolnovel.com/the-legendary-mechanic-",i]

# requests to page
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:99.0) Gecko/20100101 Firefox/99.0'}
site = requests.get(url=''.join(map(str, maw)), headers=headers)
print(site.status_code)
page = site.content

# beatifulsoup to find the paragrephes -->jobt []
text = BeautifulSoup(page,"lxml")
jobt = text.find_all("p" ,{"style":"text-align: right;"})

"""
#conert jobt to string 
sjobt="".join(map(str, jobt))
#print(k)
"""

# creat epub book 
book = epub.EpubBook()
# set metadata
book.set_identifier('id123456')
book.set_title('the legendary mechanic')
book.set_language('ar')
book.add_author('Author Authorowski')

# create chapter
c1 = epub.EpubHtml(title='Intro', file_name='chap_01.xhtml', lang='hr')
c1.content=u''.join(map(str, jobt))

# add chapter
book.add_item(c1)

# define Table Of Contents
book.toc = (epub.Link('chap_01.xhtml', 'Introduction', 'intro'),
            (epub.Section('Simple book'),(c1, ))
           )
# add default NCX and Nav file
book.add_item(epub.EpubNcx())
book.add_item(epub.EpubNav())

# define CSS style
style = 'BODY {color: white;}'
nav_css = epub.EpubItem(uid="style_nav", file_name="style/nav.css", media_type="text/css", content=style)

# add CSS file
book.add_item(nav_css)

# basic spine
book.spine = ['nav', c1]

# write to the file
epub.write_epub('ftest.epub', book, {})
