(ns trees-from-scratch.dataset
  "Core dataset representation and manipulation functions."
  (:require [tablecloth.api :as tc]
            [tech.v3.dataset.sql :as sql]
            [clojure.string :as str]))

(defn make-dataset
  "Constructs a composite dataset map containing :columns and :types."
  [columns types]
  {:columns columns
   :types types})

(defn row-count
  "Returns the number of rows in the dataset.
   Returns 0 for an empty dataset."
  [dataset]
  (let [cols (:columns dataset)]
    (if (empty? cols)
      0
      (count (val (first cols))))))

(defn column-names
  "Returns a sequence of the column keys in the dataset."
  [dataset]
  (keys (:columns dataset)))

(defn get-column
  "Retrieves a specific column vector from the dataset."
  [dataset col-key]
  (get-in dataset [:columns col-key]))

(defn select-columns
  "Returns a new dataset containing only the specified columns, preserving type metadata."
  [dataset col-keys]
  {:columns (select-keys (:columns dataset) col-keys)
   :types   (select-keys (:types dataset) col-keys)})

(defn map-columns
  "Applies a function `f` to each column vector in the dataset.
   `f` should take a column vector and return a new column vector of the same size.
   Returns a new dataset with the transformed columns and preserved types."
  [dataset f]
  (assoc dataset :columns
         (reduce-kv (fn [m k v]
                      (assoc m k (f v)))
                    {}
                    (:columns dataset))))

(defn get-type
  "Retrieves the type metadata (:continuous or :categorical) for a column."
  [dataset col-key]
  (get-in dataset [:types col-key]))

(defn get-row
  "Extracts the i-th row from the dataset as a single map {:col1 val1 :col2 val2 ...}."
  [dataset idx]
  (reduce-kv (fn [m k v]
               (assoc m k (nth v idx)))
             {}
             (:columns dataset)))

(defn select-rows
  "Returns a new composite dataset containing only the specified `indices`.
   Preserves type metadata perfectly."
  [dataset indices]
  (assoc dataset :columns
         (into {} (for [[k v] (:columns dataset)]
                    [k (mapv #(nth v %) indices)]))))

(defn split-by-continuous
  "Splits a dataset into [left-dataset right-dataset] based on a continuous feature.
   Left dataset contains rows where val <= threshold. Right dataset contains rows where val > threshold."
  [dataset col-key threshold]
  (let [col (get-column dataset col-key)
        [indices-left indices-right]
        (reduce-kv (fn [[l r] idx val]
                     (if (<= val threshold)
                       [(conj l idx) r]
                       [l (conj r idx)]))
                   [[] []]
                   col)]
    [(select-rows dataset indices-left)
     (select-rows dataset indices-right)]))

(defn split-by-categorical
  "Splits a dataset into [left-dataset right-dataset] based on a categorical feature.
   Left dataset contains rows where val == category. Right dataset contains rows where val != category."
  [dataset col-key category]
  (let [col (get-column dataset col-key)
        [indices-left indices-right]
        (reduce-kv (fn [[l r] idx val]
                     (if (= val category)
                       [(conj l idx) r]
                       [l (conj r idx)]))
                   [[] []]
                   col)]
    [(select-rows dataset indices-left)
     (select-rows dataset indices-right)]))

(defn from-tablecloth
  "Converts a tech.ml.dataset into a trees-from-scratch.dataset."
  [ds]
  (reduce (fn [acc col-name]
            (assoc acc (keyword col-name) (vec (ds col-name))))
          {}
          (tc/column-names ds)))

(defn from-csv
  "Reads a CSV using Tablecloth's robust parser, then converts it to trees-from-scratch.dataset."
  [filepath]
  (-> filepath
      (tc/dataset)
      (from-tablecloth)))

(defn from-parquet
  "Reads a Parquet file and converts it to trees-from-scratch.dataset."
  [filepath]
  (-> filepath tc/dataset from-tablecloth))

(defn from-json
  "Reads a JSON file containing an array of objects."
  [filepath]
  (-> filepath tc/dataset from-tablecloth))

(defn from-sql
  "Executes a SQL query and loads the result set."
  [db-spec query]
  (-> (sql/sql->dataset db-spec query)
      from-tablecloth))

(defn to-tablecloth
  "Converts a trees-from-scratch.dataset into a Tablecloth dataset."
  [custom-ds]
  (tc/dataset custom-ds))

(defn- validate-extension!
  "Throws an exception if the filepath does not end with the given extension."
  [filepath ext]
  (when-not (str/ends-with? filepath ext)
    (throw (IllegalArgumentException. (str "Filepath must end with " ext)))))

(defn to-file
  "Writes a trees-from-scratch.dataset to a file, inferring the format from the file extension."
  [custom-ds filepath]
  (-> custom-ds
      (to-tablecloth)
      (tc/write! filepath)))

(defn to-csv
  "Writes a trees-from-scratch.dataset to a CSV file. Validates that the filepath ends with .csv"
  [custom-ds filepath]
  (validate-extension! filepath ".csv")
  (to-file custom-ds filepath))

(defn to-json
  "Writes our custom dataset to a JSON file. Validates that the filepath ends with .json"
  [custom-ds filepath]
  (validate-extension! filepath ".json")
  (to-file custom-ds filepath))

(defn to-parquet
  "Writes a trees-from-scratch.dataset to a Parquet file. Validates that the filepath ends with .parquet"
  [custom-ds filepath]
  (validate-extension! filepath ".parquet")
  (to-file custom-ds filepath))
