# -*- coding: utf-8 -*-
# Transform web pages into a graph to analyze

from urllib.request import Request, urlopen
from urllib.error import URLError
import html.parser

__author__ = 'temigo'


class Page:
    """
    Base class for webpages
    """
    def __init__(self, url):
        self.url = url
        self.content = None

    def get_url(self):
        req = Request(self.url)
        try:
            response = urlopen(req)
        except URLError as e:
            if hasattr(e, 'reason'):
                print('We failed to reach a server.')
                print('Reason: ', e.reason)
            elif hasattr(e, 'code'):
                print('The server couldn\'t fulfill the request.')
                print('Error code: ', e.code)
        else:
            # everything is fine
            html = response.read()
            return html

    def open(self):
        self.content = str(self.get_url())  # FIXME str ?

    def detect_content(self):
        pass

    def detect_links(self, content):
        raise NotImplementedError

    def detect_sources(self, content):
        raise NotImplementedError


class Parser(html.parser.HTMLParser):
    """
    To parse pages (eg detect links)
    """
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            attributes = dict(attrs)
            if 'href' in attributes:
                self.links.append(attributes['href'])


class NewsPage(Page):
    """
    Specific class for News webpages
    """
    def detect_links(self, content):
        parser = Parser()
        parser.feed(content)
        return parser.links

    def detect_sources(self, content):
        pass

if __name__ == "__main__":
    p = NewsPage("http://fr.sputniknews.com/international/20150331/1015401831.html")
    p.open()
    p.detect_links(p.content)