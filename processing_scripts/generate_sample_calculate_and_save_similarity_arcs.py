import pymongo
import pickle
import datetime
from tqdm import tqdm
from time import time

from sklearn.metrics.pairwise import cosine_similarity

# SAVE PICKLE
def save_as_pkl_file(bugs, filename='sample_bug_reports_ids_final.pkl'):
    with open(f'sample/{filename}', 'wb') as f:
        pickle.dump(bugs, f)
        f.close()

# CALCULATE DISTANCE FUNCTIONS

def calculate_cos_similarity_tfidf(vector_a, vector_b):
    return cosine_similarity(vector_a, vector_b)[0][0].item()

def calculate_cos_similarity_bert(vector_a, vector_b):
    return cosine_similarity([vector_a], [vector_b])[0][0].item()

def calculate_categoric_similarity(from_bug, to_bug):
    multiplier = 0
    if from_bug["product"] == to_bug["product"]:
        multiplier += 0.5
    if from_bug["component"] == to_bug["component"]:
        multiplier += 0.5
    return multiplier

# MONGO INTERACTIONS FUNCTIONS

def get_mongo_conn(MONGO_URL, MONGO_DATABASE):
    client = pymongo.MongoClient(MONGO_URL)
    db = client[MONGO_DATABASE]
    return db

def retrieve_sample(db, qty, time_scope):
    db_bugs = db["bug"]

    db_bugs_query = db_bugs.aggregate([{
        "$match": {
            "tfidf_vector":      { "$exists": True },
            "embeddings_vector": { "$exists": True },
            "creation_time": {
                "$lt": time_scope["creation_time_end"], # convert to iso
                "$gt": time_scope["creation_time_start"] # convert to iso
            },
            "sample_set": {
                "$exists": False
            }
        }
    }, {
        "$sample": { "size": qty }
    }])

    bugs = []

    for b in db_bugs_query:
        b["tfidf_vector"]      = deconvert_from_mongo(b["tfidf_vector"])
        b["embeddings_vector"] = deconvert_from_mongo(b["embeddings_vector"])
        bugs.append(b)

    return bugs

def retrieve_candidates_query(db, query):
    db_bugs = db["bug"]

    db_bugs_query = db_bugs.find({
        "tfidf_vector":      { "$exists": True },
        "embeddings_vector": { "$exists": True },
        "creation_time": {
            "$lt": query["creation_time"],
        },
        "when_changed_to_resolved": {
            "$gt": query["creation_time"]
        },
        "$or": [{
            "component": query["component"],
        },
        {
            "product": query["product"]
        }]
    })

    bugs = []

    for b in db_bugs_query:
        b["tfidf_vector"]      = deconvert_from_mongo(b["tfidf_vector"])
        b["embeddings_vector"] = deconvert_from_mongo(b["embeddings_vector"])
        bugs.append(b)

    return bugs


def save_arcs(db, arcs):
    db["arc"].insert_many(arcs)

# PICKLE OPERATIONS

def deconvert_from_mongo(bin):
    return pickle.loads(bin)

# ARC CALCULATIONS

def calculate_distance_arcs_between_reports(query, others):
    arcs = []

    for other in others:
        arc = {
            "from": query["bg_number"],
            "to": other["bg_number"],
            "cos_similarity_tfidf": calculate_cos_similarity_tfidf(query["tfidf_vector"], other["tfidf_vector"]),
            "cos_similarity_word_embeddings": calculate_cos_similarity_bert(query["embeddings_vector"], other["embeddings_vector"]),
            "categoric_similarity": calculate_categoric_similarity(query, other)
        }

        arcs.append(arc)
    return arcs

def check_sample(sample_bugs, sample_info_filename):
    info = {
        "years": {},
        "products": {},
        "components": {}
    }

    # count years dist
    for sp in sample_bugs:
        # years count
        sp_year = sp["creation_time"].year
        if sp_year in info["years"].keys():
            info["years"][sp_year] += 1
        else:
            info["years"][sp_year] = 1
        
        # products count
        sp_product = sp["product"]
        if sp_product in info["products"].keys():
            info["products"][sp_product] += 1
        else:
            info["products"][sp_product] = 1
        
        # components count
        sp_component = sp["component"]
        if sp_component in info["components"].keys():
            info["components"][sp_component] += 1
        else:
            info["components"][sp_component] = 1
    

    print(f'check years: {info["years"]}')

    input()

    print(f'saving sample info on {sample_info_filename}')
    save_as_pkl_file(info, sample_info_filename)

def main():
    db = get_mongo_conn(MONGO_URL="mongodb://localhost:27017/",
                        MONGO_DATABASE="bug_report_colab")

    SAMPLE_SIZE = 10000
    SAMPLE_CREATION_DATE_FROM = datetime.datetime(2009, 1, 1, 0, 0, 0, 0) # converter para iso 
    SAMPLE_CREATION_DATE_TO   = datetime.datetime(2012, 12, 31, 23, 59, 59, 0) # converter para iso
    SAMPLE_FILENAME = 'sample_bug_reports_final_180123.pkl'

    print("Retrieving sample...")
    sample_bugs = retrieve_sample(db, SAMPLE_SIZE, {
        "creation_time_start": SAMPLE_CREATION_DATE_FROM,
        "creation_time_end": SAMPLE_CREATION_DATE_TO
    })

    check_sample(sample_bugs, "QUICK_INFORMATIONS_"+SAMPLE_FILENAME)

    sample_bugs_ids = [sp["bg_number"] for sp in sample_bugs]

    print(f'saving ids sample on pkl...')
    save_as_pkl_file(sample_bugs_ids, SAMPLE_FILENAME)

    print("calculating and saving arcs...")
    total_time_a = time()
    for qb in tqdm(sample_bugs):

        candidates = retrieve_candidates_query(db=db, query=qb)

        qb_arcs = calculate_distance_arcs_between_reports(qb, candidates)

        if (len(qb_arcs) != 0):
            print(f'saving {len(qb_arcs)} arcs from ID={qb["bg_number"]}...')
            save_arcs(db, qb_arcs)
        else:
            print(f'no candidates for ID={qb["bg_number"]}')

    total_time_in_ms = int((time() - total_time_a) * 1000)
    print(f"Total time to calculate and save all arcs from {len(sample_bugs)} bugs: {total_time_in_ms}ms -> {total_time_in_ms/1000}s")

if __name__ == '__main__':
    main()