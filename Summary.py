# Loaders
from langchain.schema import Document
from langchain import PromptTemplate
# Splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Model
from langchain.chat_models import ChatOpenAI

# Embedding Support

from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
# Summarizer we'll use for Map Reduce
from langchain.chains.summarize import load_summarize_chain

# Data Science
import numpy as np
from sklearn.cluster import KMeans


load_dotenv()

def split_text(text):

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=9000,
                                              chunk_overlap=3000)
    chunk = text_splitter.split_text(text)
    docs = text_splitter.create_documents([text])
    # print(len(docs))
    # print(llm.get_num_tokens(chunk[0]))
    embeddings = OpenAIEmbeddings()
    vectors = embeddings.embed_documents([x.page_content for x in docs])
    num_clusters = min(len(vectors),6)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    closest_indices = []

    # Loop through the number of clusters you have
    for i in range(num_clusters):
        # Get the list of distances from that particular cluster center
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)

        # Find the list position of the closest one (using argmin to find the smallest distance)
        closest_index = np.argmin(distances)

        # Append that position to your closest indices list
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)
    # print(selected_indices)
    llm3 = ChatOpenAI(temperature=0,
                      max_tokens=1000,
                      model='gpt-3.5-turbo'
                      )

    map_prompt = """
    You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
    Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
    Your response should be at least three paragraphs and fully encompass what was said in the passage.

    ```{text}```
    FULL SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    map_chain = load_summarize_chain(llm=llm3,
                                     chain_type="stuff",
                                     prompt=map_prompt_template)
    selected_docs = [docs[doc] for doc in selected_indices]

    # Make an empty list to hold your summaries
    summary_list = []

    # Loop through a range of the lenght of your selected docs
    for i, doc in enumerate(selected_docs):
        chunk_summary = map_chain.run([doc])

            # Append that summary to your list
        summary_list.append(chunk_summary)

        # print(f"Summary #{i} (chunk #{selected_indices[i]}) - Preview: {chunk_summary[:250]} \n")


    summaries = "\n".join(summary_list)

    # Convert it back to a document
    summaries = Document(page_content=summaries)
    llm4 = ChatOpenAI(temperature=0,
                      max_tokens=1000,
                      model='gpt-3.5-turbo',
                      request_timeout=120
                      )
    combine_prompt = """
    You will be given a series of summaries from a book. The summaries will be enclosed in triple backticks (```)
    Your goal is to give a verbose summary of what happened in the story.
    The reader should be able to grasp what happened in the book.

    ```{text}```
    VERBOSE SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    reduce_chain = load_summarize_chain(llm=llm4,
                                        chain_type="stuff",
                                        prompt=combine_prompt_template
                                        )
    output = reduce_chain.run([summaries])
    return output