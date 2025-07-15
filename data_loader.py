from langchain_community.utilities import WikipediaAPIWrapper

def load_wikipedia(query):
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)
