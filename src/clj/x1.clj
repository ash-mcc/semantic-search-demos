(ns x1
  "This demos the setup for Question-Answer-Retrieval.
   You can input a query or a question. 
   The script then uses semantic search to find relevant passages in Simple English Wikipedia.
   This has been derived from: https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic-search/semantic_search_wikipedia_qa.py
   
   Some background on SentenceTransformers ( https://www.sbert.net )...
   
   SentenceTransformers is a Python framework for state-of-the-art sentence, text and image embeddings. 
   The initial work is described in our paper Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
   ( https://arxiv.org/abs/1908.10084 ).
   
   You can use this framework to compute sentence / text embeddings for more than 100 languages. 
   These embeddings can then be compared e.g. with cosine-similarity to find sentences with a similar meaning. 
   This can be useful for semantic textual similar, semantic search, or paraphrase mining.
   
   The framework is based on PyTorch and Transformers ( https://huggingface.co/transformers/ ) 
   and offers a large collection of pre-trained models ( https://www.sbert.net/docs/pretrained_models.html )
   tuned for various tasks. Further, it is easy to fine-tune your own models ( https://www.sbert.net/docs/training/overview.html )."
  (:require [clojure.java.io :as io]
            [clojure.data.json :as json]
            [scicloj.ml.dataset :as ds]
            [libpython-clj2.python :as py]
            [libpython-clj2.require :refer [require-python]])
  (:import java.util.zip.GZIPInputStream))


(require-python '[builtins]
                '[sentence_transformers :as tf]
                '[torch])



(defn ensure-local-file
  [f src-url]
  (when (not (.exists f))
    (with-open [is (io/input-stream (io/as-url src-url))
                os (io/output-stream f)]
      (io/copy is os))))


;; As model, we use: nq-distilbert-base-v1
;; It was trained on the Natural Questions dataset, 
;; a dataset with real questions from Google Search
;; together with annotated data from Wikipedia providing the answer. 
(def model-name "nq-distilbert-base-v1")


;; use the Bi-Encoder to encode all passages, so that we can use it with semantic search
(def bi-encoder (tf/SentenceTransformer model-name :cache_folder "data"))
(builtins/type bi-encoder) ;; => sentence_transformers.SentenceTransformer.SentenceTransformer



;; As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
;; about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder.
(def wikipedia-file (io/file "data/simplewiki-2020-11-01.jsonl.gz"))
(ensure-local-file wikipedia-file "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/simplewiki-2020-11-01.jsonl.gz")

(def passages
  (with-open [rdr (-> wikipedia-file io/input-stream GZIPInputStream. io/reader)]
    (doall
     (apply concat
            (for [line (line-seq rdr)]
              (let [json       (json/read-str line :key-fn keyword)
                    title      (:title json)
                    paragraphs (:paragraphs json)]
                (for [paragraph paragraphs]
                  [title paragraph])))))))

(count passages) ;; => 509663

;; To speed things up, pre-computed embeddings are downloaded.
;; The provided file encoded the passages with the model 'nq-distilbert-base-v1'

(def corpus-embeddings
  (if (= "nq-distilbert-base-v1" model-name)
    (let [embeddings-file (io/file "data/simplewiki-2020-11-01-nq-distilbert-base-v1.pt")] 
      (ensure-local-file embeddings-file "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/simplewiki-2020-11-01-nq-distilbert-base-v1.pt")
      (let [embeddings (-> (.getPath embeddings-file)
                           (torch/load :map_location (torch/device "cpu"))
                           (py/py. "float") ;; Convert embedding file to float
                           )]
        (if (py/py. torch/cuda "is_available")
          (py/py. embeddings "to" "cuda") 
          embeddings)))
    (py/py. bi-encoder "encode" passages :convert_to_tensor true :show_progress_bar true) ;; Here, we compute the corpus_embeddings from scratch (which can take a while depending on the GPU)
    ))

(builtins/type corpus-embeddings) ;; => torch.Tensor






(def query "What country is beside Sweden?")

;; Encode the query using the bi-encoder and find potentially relevant passages
(def start-time (System/currentTimeMillis))
(def question-embedding (py/py. bi-encoder "encode" query :convert_to_tensor true))
(def top-k 5) ;; Number of passages we want to retrieve with the bi-encoder
(def hits (-> (py/py. tf/util "semantic_search" question-embedding corpus-embeddings :top_k top-k)
              first ;; 1 query so get just the 1st result
              py/->jvm)) 
(def end-time (System/currentTimeMillis))

;; Output of top-k hits
(format "%.3f seconds" (double (/ (- end-time start-time) 1000)))
(-> (ds/dataset hits)
    (ds/add-column :passage (fn [ds]
                               (map #(nth passages %)
                                    (get ds "corpus_id"))))
    (ds/separate-column :passage [:passage-title :passage-content] identity))

