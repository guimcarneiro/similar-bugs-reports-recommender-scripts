import pymongo

from recommender import SimilarBugReportsRecommendationSystem
from data_loader import EnhancedMongoDataLoader

from time import time

DATABASE_URL = "mongodb://localhost:27017/"
DATABASE_NAME = 'bug_report_colab'

# CONEXÃO COM MONGODB

def get_mongo_conn(MONGO_URL, MONGO_DATABASE):
    client = pymongo.MongoClient(MONGO_URL)
    db = client[MONGO_DATABASE]
    return db

# CÁLCULO DE MÉTRICAS DE AVALIAÇÃO

def positive_result(query, result):
    return (query["assigned_to"] == result["assigned_to"])

# Num de recomendações retornadas por uma dada query
# É a porcentagem de queries com pelo menos K recomendações
def calculate_feedback(query, results, K):
    if  (len(results) >= K):
        return 1.0
    return 0.0

# Mede a razão de recomendações relevantes.
# Mais especificamente, define-se como a precisão das primeiras K recomendações.
def calculate_precision(query, results):
    relevants = 0.0
    for result in results:
        if positive_result(query, result["item"]):
            relevants += 1.0
    
    if len(results) > 0:
        return (relevants/len(results))
    else:
        return 0

# Métrica comum para responder sobre a utilidade das recomendações
# dentro das top-K recomendações.
def calculate_likelihood(query, results):
    for result in results:
        if positive_result(query, result["item"]):
            return 1.0
    return 0.0

def calculate_avg_metric(results):
    total = 0.0
    for res in results:
        total += res

    if len(results) > 0:
        return (total/len(results))
    else:
        return 0.0

def retrieve_sample(db):
    db_bugs = db["bug"]

    sample = []

    bugs = db_bugs.find({
        "sample_set": True
    }, {
        "bg_number": True,
        "summary": True,
        "assigned_to": True
    })

    for bug in bugs:
        sample.append(bug)
    
    return sample

def save_result_row(db, result):
    db_results = db["result"]

    db_results.insert_one({
        "version": result["version"],
        "feedback": result["feedback"],
        "precision": result["precision"],
        "likelihood": result["likelihood"],
        "query": result["query"],
        "recommendations": result["recommendations"],
        "k": result["k"]
    })

def print_recommendations_resumee(recommendations):
    for i, r in enumerate(recommendations):
        print(f'{i+1} - {r["bg_number"]},rlv={int(r["relevant"])}, scr={r["score"]}, comp={r["component"]}, prod={r["product"]}, summ={r["summary"]}')

def print_results_resumee(results):
    print(f'[RESULT] ID={results["query"]} - fb={results["feedback"]} prc={results["precision"]} lkh={results["likelihood"]}')
    #for x in results["recommendations"]

def instanciate_recommender():
    data_loader = EnhancedMongoDataLoader(database=DATABASE_NAME, host='localhost', port=27017)
    recommender = SimilarBugReportsRecommendationSystem(data_loader=data_loader)

    return recommender

def execute_evaluation(db, k, evaluation_version, simi_score_type, save_results=False):
    print('instanciating recommender...')
    recommender = instanciate_recommender()

    print('retrieving sample...')
    sample = retrieve_sample(db)

    results = []

    for sample_bug in sample:
        print(f'[REQUEST]Requesting recommendations for ID={sample_bug["bg_number"]}, Summary={sample_bug["summary"]}')
        recommendations = recommender.get_recommendations(query=sample_bug, K=k, similarity_score_type=simi_score_type)
        result = {
            "version": evaluation_version,
            "query": sample_bug["bg_number"],
            "feedback": calculate_feedback(query=sample_bug, results=recommendations, K=k),
            "precision": calculate_precision(query=sample_bug, results=recommendations),
            "likelihood": calculate_likelihood(query=sample_bug, results=recommendations),
            "recommendations": [
                {
                    "bg_number": rec["item"]["bg_number"],
                    "summary": rec["item"]["summary"],
                    "product": rec["item"]["product"],
                    "component": rec["item"]["component"],
                    "score": rec["score"],
                    "cos_similarity_tfidf": rec["cos_similarity_tfidf"],
                    "cos_similarity_word_embeddings": rec["cos_similarity_word_embeddings"],
                    "relevant": positive_result(sample_bug, rec["item"]),
                } for rec in recommendations
            ],
            "k": k
        }

        if save_results:
            print(f'saving result row for ID={sample_bug["bg_number"]}')
            save_result_row(db, result)
        else:
            print_recommendations_resumee(result["recommendations"])
            #print_results_resumee(result)

        results.append(result)
    
    avg_feedback   = calculate_avg_metric(results=[ r["feedback"] for r in results ])
    avg_precision  = calculate_avg_metric(results=[ r["precision"] for r in results ])
    avg_likelihood = calculate_avg_metric(results=[ r["likelihood"] for r in results ])

    print('-'*50)

    print(f'Metrics resumee of the evaluation were:')
    print(f'Avg feedback: {avg_feedback}')
    print(f'Avg precision: {avg_precision}')
    print(f'Avg likelihood: {avg_likelihood}')
    
if __name__ == '__main__':
    print('creating db connection...')
    db = get_mongo_conn(MONGO_URL=DATABASE_URL,
                        MONGO_DATABASE=DATABASE_NAME)

    for version in ['categoric_tfidf_we', 'categoric_tfidf', 'categoric_we']:
        for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
            time_a = time()
            execute_evaluation(db=db, k=i, evaluation_version=version, simi_score_type=version, save_results=False)

            final_time = time() - time_a
            print(f'total evaluation time: {final_time}s : {version} : K={i}')