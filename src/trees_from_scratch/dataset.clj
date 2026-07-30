(ns trees-from-scratch.dataset
  "Core dataset representation and manipulation functions.")

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
