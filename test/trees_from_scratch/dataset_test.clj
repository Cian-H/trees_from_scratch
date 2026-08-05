(ns trees-from-scratch.dataset-test
  (:require [clojure.test :refer [deftest is testing]]
            [trees-from-scratch.test-utils :refer [with-temp-file]]
            [trees-from-scratch.dataset :as ds]
            [tech.v3.dataset.sql :as sql]))

(def classification-dataset
  {:columns {:age    [15 22 35 45 50]
             :income [20 25 70 90 85]
             :label  ["No" "No" "Yes" "Yes" "Yes"]}
   :types   {:age    :continuous
             :income :continuous
             :label  :categorical}})

(def categorical-dataset
  {:columns {:outlook ["sunny" "sunny" "overcast" "rain"]
             :play    ["no" "no" "yes" "yes"]}
   :types   {:outlook :categorical
             :play    :categorical}})

(deftest test-dataset-creation
  (testing "creating dataset manually with make-dataset"
    (let [d (ds/make-dataset {:a [1]} {:a :continuous})]
      (is (= {:a [1]} (:columns d)))
      (is (= {:a :continuous} (:types d))))))

(deftest test-dataset-utils
  (testing "calculating row count of a dataset"
    (is (= 5 (ds/row-count classification-dataset)))
    (is (= 0 (ds/row-count {:columns {}}))))

  (testing "getting column names"
    (is (= #{:age :income :label} (set (ds/column-names classification-dataset)))))

  (testing "getting a specific column"
    (is (= [15 22 35 45 50] (ds/get-column classification-dataset :age)))
    (is (nil? (ds/get-column classification-dataset :non-existent))))

  (testing "getting column type"
    (is (= :continuous (ds/get-type classification-dataset :age)))
    (is (= :categorical (ds/get-type classification-dataset :label))))

  (testing "extracting a single row map by index"
    (is (= {:age 15 :income 20 :label "No"} (ds/get-row classification-dataset 0)))
    (is (= {:age 35 :income 70 :label "Yes"} (ds/get-row classification-dataset 2))))

  (testing "selecting subset of rows by index vector preserves types"
    (let [subset (ds/select-rows classification-dataset [0 2])]
      (is (= {:age [15 35] :income [20 70] :label ["No" "Yes"]} (:columns subset)))
      (is (= 2 (ds/row-count subset)))
      (is (= (:types classification-dataset) (:types subset)))))

  (testing "selecting subset of columns preserves structure and types"
    (let [subset (ds/select-columns classification-dataset [:age :label])]
      (is (= #{:age :label} (set (ds/column-names subset))))
      (is (= {:age :continuous :label :categorical} (:types subset)))
      (is (= [15 22 35 45 50] (ds/get-column subset :age)))
      (is (= ["No" "No" "Yes" "Yes" "Yes"] (ds/get-column subset :label)))))

  (testing "mapping a function over all columns"
    (let [mapped (ds/map-columns classification-dataset (fn [col] (mapv str col)))]
      (is (= ["15" "22" "35" "45" "50"] (ds/get-column mapped :age)))
      (is (= ["20" "25" "70" "90" "85"] (ds/get-column mapped :income)))
      ;; the function maps over all columns, even target variables
      (is (= ["No" "No" "Yes" "Yes" "Yes"] (ds/get-column mapped :label)))
      ;; types are preserved structurally
      (is (= (:types classification-dataset) (:types mapped)))))

  (testing "adding a column updates column data and infers/sets type metadata"
    (let [added (ds/add-column classification-dataset :score [10 20 30 40 50])
          added-cat (ds/add-column classification-dataset :group ["A" "A" "B" "B" "C"] :categorical)]
      (is (= [10 20 30 40 50] (ds/get-column added :score)))
      (is (= :continuous (ds/get-type added :score)))
      (is (= ["A" "A" "B" "B" "C"] (ds/get-column added-cat :group)))
      (is (= :categorical (ds/get-type added-cat :group)))))

  (testing "dropping a column or set of columns removes them from columns and types"
    (let [dropped-single (ds/drop-column classification-dataset :income)
          dropped-multi  (ds/drop-column classification-dataset [:income :label])]
      (is (= #{:age :label} (set (ds/column-names dropped-single))))
      (is (nil? (ds/get-type dropped-single :income)))
      (is (= #{:age} (set (ds/column-names dropped-multi))))
      (is (nil? (ds/get-column dropped-multi :label)))
      (is (nil? (ds/get-type dropped-multi :label)))))

  (testing "renaming columns updates both columns and types maps"
    (let [renamed (ds/rename-columns classification-dataset {:age :years-old, :income :earnings})]
      (is (= #{:years-old :earnings :label} (set (ds/column-names renamed))))
      (is (= :continuous (ds/get-type renamed :years-old)))
      (is (= :continuous (ds/get-type renamed :earnings)))
      (is (nil? (ds/get-column renamed :age)))
      (is (nil? (ds/get-type renamed :income)))))

  (testing "replacing an existing column updates values and type metadata"
    (let [replaced (ds/replace-column classification-dataset :income [100 200 300 400 500])
          replaced-type (ds/replace-column classification-dataset :income ["low" "low" "mid" "high" "high"] :categorical)]
      (is (= [100 200 300 400 500] (ds/get-column replaced :income)))
      (is (= :continuous (ds/get-type replaced :income)))
      (is (= ["low" "low" "mid" "high" "high"] (ds/get-column replaced-type :income)))
      (is (= :categorical (ds/get-type replaced-type :income))))))

(deftest test-split-dataset
  (testing "splitting dataset with continuous feature"
    (let [[left right] (ds/split-by-continuous classification-dataset :age 30)]
      (is (= 2 (ds/row-count left)))
      (is (= 3 (ds/row-count right)))
      (is (= [15 22] (ds/get-column left :age)))
      (is (= [35 45 50] (ds/get-column right :age)))
      ;; Metadata preserved
      (is (= :continuous (ds/get-type left :age)))))

  (testing "splitting dataset with categorical feature"
    (let [[left right] (ds/split-by-categorical categorical-dataset :outlook "sunny")]
      (is (= 2 (ds/row-count left)))
      (is (= 2 (ds/row-count right)))
      (is (every? #(= % "sunny") (ds/get-column left :outlook)))
      (is (every? #(not= % "sunny") (ds/get-column right :outlook)))
      ;; Metadata preserved
      (is (= :categorical (ds/get-type right :outlook))))))

(deftest round-trip-io-test
  (let [original-data {:age [25 30 42]
                       :income [50000 65000 120000]
                       :target [0 1 1]}]

    (testing "CSV Round-Trip"
      (with-temp-file [filepath ["test-data" ".csv"]]
        (ds/to-csv original-data filepath)
        (let [read-data (ds/from-csv filepath)]
          (is (= original-data read-data)))))

    (testing "JSON Round-Trip"
      (with-temp-file [filepath ["test-data" ".json"]]
        (ds/to-json original-data filepath)
        (let [read-data (ds/from-json filepath)]
          (is (= original-data read-data)))))

    (testing "Parquet Round-Trip"
      (with-temp-file [filepath ["test-data" ".parquet"]]
        (ds/to-parquet original-data filepath)
        (let [read-data (ds/from-parquet filepath)]
          (is (= original-data read-data)))))

    (testing "SQL Round-Trip (SQLite)"
      (with-temp-file [filepath ["test-data" ".sqlite"]]
        (let [jdbc-url (str "jdbc:sqlite:" filepath)
              table-name "test_data"
              tc-ds (ds/to-tablecloth original-data)]
          (with-open [conn (java.sql.DriverManager/getConnection jdbc-url)]
            (sql/create-table! conn tc-ds {:table-name table-name})
            (sql/insert-dataset! conn tc-ds {:table-name table-name})
            (let [query "SELECT * FROM test_data"
                  read-data (ds/from-sql conn query)]
              (is (= original-data read-data)))))))))
