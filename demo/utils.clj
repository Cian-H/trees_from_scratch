(ns utils
  (:require [trees-from-scratch.dataset :as ds]
            [clojure.java.io :as io]))

(defn fetch-and-cache
  [url cache-filename]
  (let [cache-dir ".data_cache"
        cache-file (str cache-dir "/" cache-filename)]
    (when-not (.exists (io/file cache-dir))
      (.mkdir (io/file cache-dir)))
    (if (.exists (io/file cache-file))
      (do
        (println (str "Loading " cache-filename " from cache..."))
        (ds/from-csv cache-file))
      (do
        (println (str "Pulling down " cache-filename "..."))
        (let [data (ds/from-csv url)]
          (println "Caching dataset...")
          (ds/to-csv data cache-file)
          data)))))

(defn fetch-iris []
  (let [raw-data (fetch-and-cache "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv" "iris.csv")]
    (ds/make-dataset raw-data
                     {:sepal_length :continuous
                      :sepal_width  :continuous
                      :petal_length :continuous
                      :petal_width  :continuous
                      :species      :categorical})))

(defn fetch-tips []
  (let [raw-data (fetch-and-cache "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv" "tips.csv")]
    (ds/make-dataset raw-data
                     {:total_bill :continuous
                      :tip        :continuous
                      :sex        :categorical
                      :smoker     :categorical
                      :day        :categorical
                      :time       :categorical
                      :size       :continuous})))

(defn train-test-split [dataset train-ratio]
  (let [total-rows (ds/row-count dataset)
        split-idx (int (* train-ratio total-rows))
        shuffled-indices (shuffle (range total-rows))
        train-indices (take split-idx shuffled-indices)
        test-indices (drop split-idx shuffled-indices)]
    {:train (ds/select-rows dataset train-indices)
     :test  (ds/select-rows dataset test-indices)}))
