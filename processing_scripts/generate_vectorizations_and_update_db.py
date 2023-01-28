import pymongo
import nltk
import pickle
import string
from bson import Binary
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfIdfVectorizer

# CÓDIGO RELACIONADO À VETORIZADORES

BERT = 'all-MiniLM-L6-v2'

class BertVectorizer():
    def __init__(self):
        print(f'instanciating bert...')
        self.vectorizer = SentenceTransformer(BERT)

    def transform(self, docs):
        vectorized_docs = [ self.vectorizer.encode(doc) for doc in docs ]
        return vectorized_docs

class TfidfVectorizer():
    def __init__(self, corpus):
        self.vectorizer = SklearnTfIdfVectorizer()
        print(f'Fitting tfidfVectorizer with a corpus with size of {len(corpus)} docs...')
        self.vectorizer.fit(corpus)
    
    def transform(self, text):
        return self.vectorizer.transform([text])
        
# CÓDIGO RELACIONADO À PRE-PROCESSAMENTO E GERAÇÃO DE VETORES

def pre_process(text):
    print('start preprocessing...')

    # remove stopwords and punctuation
    print('removing stopwords...')
    stop = set(stopwords.words('english') + list(string.punctuation))
    return " ".join([w for w in word_tokenize(text.lower()) if w not in stop])   

def generate_tfidf(pp_document, tfidf_vectorizer):
    return tfidf_vectorizer.transform(pp_document)

def generate_embeddings(document, bert_vectorizer):
    return bert_vectorizer.transform(document)[0]


# CÓDIGO RELACIONADO À MONGODB

def convert_to_mongo_acceptable(vector, vectorization="tfidf"):
    
    if vectorization == "bert":
        print(f'{vectorization}: type={type(vector)}')
        return Binary(pickle.dumps(vector, protocol=2), subtype=128)
        # ndarray to python list
        # return vector
    if vectorization == "tfidf":
        print(f'{vectorization}: type={type(vector)}')
        # csr matrix to python list of lists (LIL)
        # return vector.tolil()
        return Binary(pickle.dumps(vector, protocol=2), subtype=128)
    return -1

def deconvert_from_mongo(bin):
    return pickle.loads(bin)

def save_vectors_on_mongo(db, bug_id, tfidf_vector, bert_vector):
    print(f'salvando vetores em mongo, id={bug_id}...')
    db.update_one({
        "bg_number": bug_id
    }, {
        "$set": {
            "tfidf_vector"     : tfidf_vector,
            "embeddings_vector": bert_vector
        }
    })

def update_tfidf_vector_on_mongo(db, bug_id, tfidf_vector):
    print(f'atualizando tfidf em mongo, id={bug_id}...')
    db.update_one({
        "bg_number": bug_id
    }, {
        "$set": {
            "tfidf_vector": tfidf_vector,
        }
    })

def retrieve_bugs_without_vectors(db):
    return db.find({
        "tfidf_vector":      { "$exists": False },
        "embeddings_vector": { "$exists": False }
    })

def retrieve_bugs_with_tfidf(db):
    return db.find({
        "tfidf_vector": { "$exists": True }
    })

# OPERAÇÕES EM DATASETS

def fix_tfidf_vectors_on_dataset(database_name):
    # connect to mongodb
    print('abrindo conexão com mongodb...')
    client = pymongo.MongoClient(f"mongodb://localhost:27017/")
    db_bugs = client[database_name]["bug"]

    # find all bug reports
    print('buscando todos os bug reports que ja possuem tfidf...')
    all_bugs = [b for b in retrieve_bugs_with_tfidf(db_bugs)]

    # preprocess all bug reports and puts on dict
    print('aplicando preprocessamentos nas descrições dos bug reports que ja possuem tfidf...')
    all_preprocessed_bugs = {}
    for b in all_bugs:
        pp_bug = {
            "id": b["bg_number"],
            "pp_description": pre_process(b["description"])
        }
        all_preprocessed_bugs[b["bg_number"]] = pp_bug

    tfidf_vectorizer = TfidfVectorizer([all_preprocessed_bugs[key]["pp_description"] for key in all_preprocessed_bugs.keys()])

    for i, b in enumerate(all_bugs):
        print(f"gerando tfidf e atualizando db para id={b['bg_number']} || contagem: {i+1}/{len(all_bugs)}")
        # gera tfidf
        tfidf_vector = generate_tfidf(all_preprocessed_bugs[ b["bg_number"] ]["pp_description"], tfidf_vectorizer)

        # atualiza no banco
        update_tfidf_vector_on_mongo(db_bugs, b["bg_number"], convert_to_mongo_acceptable(tfidf_vector, "tfidf"))

def populate_vectorizations(database_name, batch_size=10000):
    # connect to mongodb
    print('abrindo conexão com mongodb...')
    client = pymongo.MongoClient(f"mongodb://localhost:27017/")
    db_bugs = client[database_name]["bug"]

    # find all bug reports
    print('buscando todos os bug reports...')
    all_bugs = [b for b in retrieve_bugs_without_vectors(db_bugs)]

    # preprocess all bug reports and puts on dict
    print('aplicando preprocessamentos nas descrições de todos os bug reports...')
    all_preprocessed_bugs = {}
    for b in all_bugs:
        pp_bug = {
            "id": b["bg_number"],
            "pp_description": pre_process(b["description"])
        }
        all_preprocessed_bugs[b["bg_number"]] = pp_bug

    # instanciate vectorizers
    tfidf_vectorizer = TfidfVectorizer([all_preprocessed_bugs[key]["pp_description"] for key in all_preprocessed_bugs.keys()])
    bert_vectorizer  = BertVectorizer()

    all_bugs = all_bugs[:batch_size]
    while (len(all_bugs) > 0):

        print('baixando stopwords e pontuação...')
        nltk.download('stopwords')
        nltk.download('punkt')

        # generate vectors
        print('gerando vetores tfidf e embeddings para todos os bug reports...')
        bugs_context_vectors = []
        for i, b in enumerate(tqdm(all_bugs)):
            print(f'gerando vetores para id={b["bg_number"]}...')
            b_vectors = {
                "id"        : b["bg_number"],
                "embeddings": generate_embeddings(b["description"], bert_vectorizer),
                "tfidf"     : generate_tfidf(all_preprocessed_bugs[ b["bg_number"] ]["pp_description"], tfidf_vectorizer)
            }
            bugs_context_vectors.append(b_vectors)

            # a cada 50 bugs com vetores gerados, salva no BD e limpa memória
            if (len(bugs_context_vectors) == 50):
                #print('salvando em bd com vetores...')
                for b in bugs_context_vectors:
                    tfidf_vector = convert_to_mongo_acceptable(vector=b["tfidf"], vectorization="tfidf")
                    embeddings_vector = convert_to_mongo_acceptable(vector=b["embeddings"], vectorization="bert")
                    
                    save_vectors_on_mongo(
                        db=db_bugs,
                        bug_id=b["id"],
                        tfidf_vector=tfidf_vector,
                        bert_vector=embeddings_vector
                    )
                print(f"salvou +50... {i+1} de {len(all_bugs)}")
                # limpa bugs_context_vectors de 50 em 50 para não ocupar tanta memória
                bugs_context_vectors = []
    
        # 
        all_bugs = [b for b in retrieve_bugs_without_vectors(db_bugs)]
        all_bugs = all_bugs[:batch_size]


# testa desconversão de binário para tipos específicos
def testing_vectors_retrieval():
    DATABASE = "bug_reports_db"
    
    # connect to mongodb
    print('abrindo conexão com mongodb...')
    client = pymongo.MongoClient(f"mongodb://localhost:27017/")
    db_bugs = client[DATABASE]["bug"]

    test_bug = db_bugs.find_one()

    tfidf_vector = deconvert_from_mongo(test_bug["tfidf_vector"])
    bert_vector  = deconvert_from_mongo(test_bug["embeddings_vector"])

    print(f"tfidf_type deconverted: {type(tfidf_vector)} !! (must be csr_matrix)")
    print(f"bert_vector type deconverted: {type(bert_vector)} !! (must be ndarray)")

def test_retrieve_vectors_tfidf():
    DATABASE = "bug_reports_db"
    
    # connect to mongodb
    print('abrindo conexão com mongodb...')
    client = pymongo.MongoClient(f"mongodb://localhost:27017/")
    db_bugs = client[DATABASE]["bug"]

    x = [y for y in retrieve_bugs_with_tfidf(db=db_bugs)]

    print(f"total reports with tfidf: {len(x)}")

if __name__ == '__main__':
    populate_vectorizations("bug_report_colab")
    #fix_tfidf_vectors_on_dataset("bug_report_colab")
    #testing_vectors_retrieval()
    #test_retrieve_vectors_tfidf()
