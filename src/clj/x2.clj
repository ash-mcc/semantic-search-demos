(ns x2
  "This demos the setup for Retrieve & Re-Rank.
   You can input a query or a question. 
   The script then uses semantic search to find relevant passages in Simple English Wikipedia.
   This has been derived from: https://colab.research.google.com/github/UKPLab/sentence-transformers/blob/master/examples/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb"
  (:require [clojure.string :as str]
            [clojure.java.io :as io]
            [clojure.data.json :as json]
            [scicloj.ml.dataset :as ds]
            [libpython-clj2.python :as py]
            [libpython-clj2.require :refer [require-python]])
  (:import java.util.zip.GZIPInputStream))


(require-python '[builtins]
                '[sentence_transformers :as tf]
                '[torch]
                '[rank_bm25]
                '[sklearn.feature_extraction :as skf])



(defn ensure-local-file
  [f src-url]
  (when (not (.exists f))
    (with-open [is (io/input-stream (io/as-url src-url))
                os (io/output-stream f)]
      (io/copy is os))))



;; We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
(def bi-encoder (tf/SentenceTransformer "multi-qa-MiniLM-L6-cos-v1" :cache_folder "data"))
(builtins/type bi-encoder) ;; => sentence_transformers.SentenceTransformer.SentenceTransformer
(py/set-attr! bi-encoder "max_seq_length" 256) ;; Truncate long passages to 256 tokens
(def top-k 32) ;; Number of passages we want to retrieve with the bi-encoder


;; The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
(def cross-encoder (tf/CrossEncoder "cross-encoder/ms-marco-MiniLM-L-6-v2"))
(builtins/type cross-encoder) ;; => sentence_transformers.cross_encoder.CrossEncoder.CrossEncoder

;; As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
;; about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder.
(def wikipedia-file (io/file "data/simplewiki-2020-11-01.jsonl.gz"))
(ensure-local-file wikipedia-file "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/simplewiki-2020-11-01.jsonl.gz")

(def passages
  (with-open [rdr (-> wikipedia-file io/input-stream GZIPInputStream. io/reader)]
    (doall
     (for [line (line-seq rdr)]
       (let [json       (json/read-str line :key-fn keyword)
             _title     (:title json)
             paragraphs (:paragraphs json)]
                ;; use only the 1st paragraph
         (first paragraphs))))))

(count passages) ;; => 169597




(def embeddings-file (io/file "data/simplewiki-2020-11-01-multi-qa-MiniLM-L6-cos-v1.pt"))

(when (not (.exists embeddings-file))
  ;; Encode all passages into our vector space. This takes about 5 minutes (depends on your GPU speed)
  (let [embeddings (py/py. bi-encoder "encode" (py/->py-list passages) :convert_to_tensor true :show_progress_bar true)]
    (torch/save embeddings (.getPath embeddings-file))))

;; Load from a file, the pre-computed embeddings of the passages with the model 'nq-distilbert-base-v1'.
(def corpus-embeddings
  (let [embeddings (-> (.getPath embeddings-file)
                      (torch/load  :map_location (torch/device "cpu"))
                                            (py/py. "float") ;; Convert embedding file to float
                                           )]
                        (if (py/py. torch/cuda "is_available")
                          (py/py. embeddings "to" "cuda")
                          embeddings)))

(builtins/type corpus-embeddings) ;; => torch.Tensor



;; We also compare the results to lexical search (keyword search) . Here, we use
;; the BM25 algorithm which is implemented in the rank_bm25 package.

(def punctuation #"[!\"#$%&'()*+,-./:;<=>?@\[\\\]^_`{|}~]")
(def extra-spaces #"\s{2,}")
(def stopwords (set (py/get-attr skf/_stop_words "ENGLISH_STOP_WORDS")))

;; lower case the text and remove stop-words from indexing
(defn bm25-tokenised
  [s]
  (->> (str/split (str/lower-case s) #"\s")
       (map #(str/replace % punctuation ""))
       (map #(str/replace % extra-spaces ""))
       (remove stopwords)))
  
(def bm25
  (let [tokenised-corpus (map bm25-tokenised passages)]
    (rank_bm25/BM25Okapi (py/->py-list tokenised-corpus))))

(builtins/type bm25) ;; => rank_bm25.BM25Okapi





(defn search
  [query]
  (let [
        ;; BM25 search (lexical search) 
        bm25-scores (py/py. bm25 "get_scores" (bm25-tokenised query))
        bm25-hits-ds (->> bm25-scores
                          (map-indexed vector)
                          (sort-by second)
                          reverse
                          (take top-k)
                          (map (fn [[idx bm25-score]] {"corpus_id"   idx
                                                       :bm25-score bm25-score
                                                       :passage    (nth passages idx)}))
                          ds/dataset)
        
        ;; note that the `bm25-hits` list is separate from the `hits` list below 
        
        ;; Semantic Search 
        ;; Encode the query using the bi-encoder and find potentially relevant passages
        question-embedding (py/py. bi-encoder "encode" query :convert_to_tensor true)
        hits               (-> (py/py. tf/util "semantic_search" question-embedding corpus-embeddings :top_k top-k) ;; just the top k 
                               (py/get-item 0)) ;; 1 query so get just the 1st result
        
        ;; Re-Ranking 
        ;; Now, score all (k) retrieved passages with the cross-encoder
        cross-input        (for [hit hits]
                             [query (nth passages (get hit "corpus_id"))])
        cross-scores       (py/py. cross-encoder "predict" (py/->py-list cross-input))
        
        semantic-hits-ds         (-> (ds/dataset (py/->jvm hits))
                               (ds/add-column :passage (fn [ds]
                                                         (map #(nth passages %)
                                                              (get ds "corpus_id"))))
                               (ds/add-column :cross-score cross-scores)
                               (ds/add-column :bm25-score bm25-scores))
        combined-ds (-> bm25-hits-ds
                        (ds/full-join semantic-hits-ds "corpus_id")
                        (ds/reorder-columns "corpus_id" [:bm25-score "score" :cross-score :passage]))]
    
    (println (-> combined-ds
                 (ds/order-by :bm25-score :desc)
                 (ds/head 3)
                 (ds/set-dataset-name "Top-3 lexical search (BM25) hits")))
    (println (-> combined-ds
                 (ds/order-by "score" :desc)
                 (ds/head 3)
                 (ds/set-dataset-name "Top-3 Bi-Encoder Retrieval hits")))
    (println (-> combined-ds
                 (ds/order-by :cross-score :desc)
                 (ds/head 3)
                 (ds/set-dataset-name "Top-3 Cross-Encoder Re-ranker hits")))
    ))





(search "What is the capital of the United States?")
; =>
;Top-3 lexical search (BM25) hits [3 8]:
;
;| corpus_id | :bm25-score |      score | :cross-score |                                                                                                                                                                                                                                                                          :passage | right.corpus_id |                                                                                                         :right.passage | :right.bm25-score |
;|----------:|------------:|-----------:|-------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|------------------------------------------------------------------------------------------------------------------------|------------------:|
;|    121824 | 13.31842493 |            |              | Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states. The federal government (including the United States military) also uses capital punishment. |                 |                                                                                                                        |                   |
;|      6642 | 11.43451041 | 0.48754507 |   0.31415635 |                                                                                                                                                            Ohio is one of the 50 states in the United States. Its capital is Columbus. Columbus also is the largest city in Ohio. |            6642 | Ohio is one of the 50 states in the United States. Its capital is Columbus. Columbus also is the largest city in Ohio. |        0.00000000 |
;|     41385 | 11.18038457 | 0.48217300 |   1.75778651 |                                                                                                                                                                  Nevada is one of the United States' states. Its capital is Carson City. Other big cities are Las Vegas and Reno. |           41385 |       Nevada is one of the United States' states. Its capital is Carson City. Other big cities are Las Vegas and Reno. |        3.32310493 |
;
;Top-3 Bi-Encoder Retrieval hits [3 8]:
;
;| corpus_id | :bm25-score |      score | :cross-score | :passage | right.corpus_id |                                                                                                                                                                                                                                       :right.passage | :right.bm25-score |
;|----------:|------------:|-----------:|-------------:|----------|----------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------:|
;|           |             | 0.62162197 |  -7.18850136 |          |          111858 |                                                                                                                                                                                                                         Cities in the United States: |               0.0 |
;|           |             | 0.59690481 |   3.68094182 |          |           37190 | The United States Capitol is the building where the United States Congress meets. It is the center of the legislative branch of the U.S. federal government. It is in Washington, D.C., on top of Capitol Hill at the east end of the National Mall. |               0.0 |
;|           |             | 0.59573293 |  -5.12398148 |          |           63510 |                                                                                                                                                                                                                                In the United States: |               0.0 |
;
;Top-3 Cross-Encoder Re-ranker hits [3 8]:
;
;| corpus_id | :bm25-score |      score | :cross-score |                                                                                                                                                                                                                                                                                                                            :passage | right.corpus_id |                                                                                                                                                                                                                                                                                                                                                  :right.passage | :right.bm25-score |
;|----------:|------------:|-----------:|-------------:|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------:|
;|     59995 |  9.34345863 | 0.56983256 |   8.90580082 | Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district. The President of the USA and many major national government offices are in the territory. This makes it the political center of the United States of America. |           59995 |                             Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district. The President of the USA and many major national government offices are in the territory. This makes it the political center of the United States of America. |               0.0 |
;|           |             | 0.54652649 |   3.75464106 |                                                                                                                                                                                                                                                                                                                                     |           14712 | A capital city (or capital town or just capital) is a city or town, specified by law or constitution, by the government of a country, or part of a country, such as a state, province or county. It usually serves as the location of the government's central meeting place and offices. Most of the country's leaders and officials work in the capital city. |               0.0 |
;|           |             | 0.59690481 |   3.68094182 |                                                                                                                                                                                                                                                                                                                                     |           37190 |                                                                                                            The United States Capitol is the building where the United States Congress meets. It is the center of the legislative branch of the U.S. federal government. It is in Washington, D.C., on top of Capitol Hill at the east end of the National Mall. |               0.0 |



(search "What is the best orchestra in the world?")
; =>
;Top-3 lexical search (BM25) hits [3 8]:
;
;| corpus_id | :bm25-score |      score | :cross-score |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 :passage | right.corpus_id |                                                                                                                         :right.passage | :right.bm25-score |
;|----------:|------------:|-----------:|-------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|----------------------------------------------------------------------------------------------------------------------------------------|------------------:|
;|     50774 | 15.34024892 |            |              | The NHK Symphony Orchestra is a Japanese orchestra based in Tokyo, Japan. In Japanese it is written: NHK交響楽団, pronounced: Enueichikei Kōkyō Gakudan. When the orchestra was started in 1926 it was called "New Symphony Orchestra". It was the first large professional orchestra in Japan. Later, it changed its name to "Japan Symphony Orchestra". In 1951 it started to get money from the Japanese radio station NHK (Nippon Hōsō Kyōkai), so it changed its name again to the name it has now. It is thought of as the best orchestra in Japan. They have played in many parts of the world, including at the BBC Proms in London. |                 |                                                                                                                                        |                   |
;|     24345 | 15.33996339 | 0.56658244 |   3.51841807 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   The BBC Symphony Orchestra is the main orchestra of the British Broadcasting Corporation. It is one of the best orchestras in Britain. |           24345 | The BBC Symphony Orchestra is the main orchestra of the British Broadcasting Corporation. It is one of the best orchestras in Britain. |               0.0 |
;|      3002 | 14.09390242 |            |              |                                                                                                                                                                                                                                                                    The Bamberger Symphoniker (Bamberg Symphony Orchestra) is a world-famous orchestra from the city of Bamberg, Germany. It was formed in 1946. Most of the musicians who formed the orchestra were Germans who had been forced to leave Czechoslovakia after the World War II. Most of them had previously been members of the German Philharmonic Orchestra of Prague. |                 |                                                                                                                                        |                   |
;
;Top-3 Bi-Encoder Retrieval hits [3 8]:
;
;| corpus_id | :bm25-score |      score | :cross-score |                                                                                                                                                                                                       :passage | right.corpus_id |                                                                                                                                                                                                 :right.passage | :right.bm25-score |
;|----------:|------------:|-----------:|-------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------:|
;|     50757 | 11.62452749 | 0.70082593 |   5.79413509 |                                      The Vienna Philharmonic (in German: die Wiener Philharmoniker) is an orchestra based in Vienna, Austria. It is thought of as one of the greatest orchestras in the world. |           50757 |                                      The Vienna Philharmonic (in German: die Wiener Philharmoniker) is an orchestra based in Vienna, Austria. It is thought of as one of the greatest orchestras in the world. |        3.58173589 |
;|           |             | 0.64091551 |  -1.20276213 |                                                                                                                                                                                                                |           53454 |                                                                                                                                                     The Vienna Symphony () is an orchestra in Vienna, Austria. |        0.00000000 |
;|     63108 | 13.61344323 | 0.63971919 |   5.32385540 | The Berlin Philharmonic (in German: Die Berliner Philharmoniker), is an orchestra from Berlin, Germany. It is one of the greatest orchestras in the world. The conductor of the orchestra is Sir Simon Rattle. |           63108 | The Berlin Philharmonic (in German: Die Berliner Philharmoniker), is an orchestra from Berlin, Germany. It is one of the greatest orchestras in the world. The conductor of the orchestra is Sir Simon Rattle. |        0.00000000 |
;
;Top-3 Cross-Encoder Re-ranker hits [3 8]:
;
;| corpus_id | :bm25-score |      score | :cross-score |                                                                                                                                                                                                       :passage | right.corpus_id |                                                                                                                                                                                                 :right.passage | :right.bm25-score |
;|----------:|------------:|-----------:|-------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------:|
;|     24383 | 11.35268310 | 0.59887135 |   5.95216894 |                         The London Symphony Orchestra (LSO) is one of the most famous orchestras of the world. They are based in London's Barbican Centre, but they often tour to lots of different countries. |           24383 |                         The London Symphony Orchestra (LSO) is one of the most famous orchestras of the world. They are based in London's Barbican Centre, but they often tour to lots of different countries. |        0.00000000 |
;|     50757 | 11.62452749 | 0.70082593 |   5.79413509 |                                      The Vienna Philharmonic (in German: die Wiener Philharmoniker) is an orchestra based in Vienna, Austria. It is thought of as one of the greatest orchestras in the world. |           50757 |                                      The Vienna Philharmonic (in German: die Wiener Philharmoniker) is an orchestra based in Vienna, Austria. It is thought of as one of the greatest orchestras in the world. |        3.58173589 |
;|     63108 | 13.61344323 | 0.63971919 |   5.32385540 | The Berlin Philharmonic (in German: Die Berliner Philharmoniker), is an orchestra from Berlin, Germany. It is one of the greatest orchestras in the world. The conductor of the orchestra is Sir Simon Rattle. |           63108 | The Berlin Philharmonic (in German: Die Berliner Philharmoniker), is an orchestra from Berlin, Germany. It is one of the greatest orchestras in the world. The conductor of the orchestra is Sir Simon Rattle. |        0.00000000 |



