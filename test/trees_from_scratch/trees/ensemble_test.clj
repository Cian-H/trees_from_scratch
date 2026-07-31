(ns trees-from-scratch.trees.ensemble-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [trees-from-scratch.trees.core :as tree-core]
            [trees-from-scratch.trees.ensemble :as ensemble]
            [trees-from-scratch.trees.binary :as binary-tree]))

(deftest test-ensemble-creation
  (testing "make creates an Ensemble record with agg and trees"
    (let [agg   identity
          t1    (binary-tree/make-leaf 1)
          ens   (ensemble/make agg t1)]
      (is (instance? trees_from_scratch.trees.ensemble.Ensemble ens))
      (is (= agg (:agg ens)))
      (is (= [t1] (:trees ens))))))

(deftest test-ensemble-predict
  (testing "ensemble predicts by aggregating sub-tree predictions"
    (let [t1    (binary-tree/make-leaf 10)
          t2    (binary-tree/make-leaf 20)
          t3    (binary-tree/make-leaf 30)
          ;; Aggregation function: sum
          agg   (fn [preds] (reduce + preds))
          ens   (ensemble/make agg t1 t2 t3)]
      ;; The leaves predict 10, 20, 30 for any row. Sum is 60.
      (is (= 60 (tree-core/predict ens {:any "row"})))))

  (testing "ensemble handles different types of aggregations (e.g. majority vote)"
    (let [t1    (binary-tree/make-leaf "A")
          t2    (binary-tree/make-leaf "B")
          t3    (binary-tree/make-leaf "A")
          ;; Aggregation function: majority vote
          agg   (fn [preds] (->> preds frequencies (sort-by val >) ffirst))
          ens   (ensemble/make agg t1 t2 t3)]
      (is (= "A" (tree-core/predict ens {}))))))

(deftest test-ensemble-tree-depth
  (testing "depth of empty ensemble is 0"
    (let [ens (ensemble/make identity)]
      (is (= 0 (tree-core/tree-depth ens)))))

  (testing "depth of ensemble is the max depth of its sub-trees"
    (let [leaf  (binary-tree/make-leaf 1)                                   ;; depth 1
          split (binary-tree/make-continuous-split :x 10 leaf leaf)         ;; depth 2
          deep  (binary-tree/make-continuous-split :y 20 split split)       ;; depth 3
          ens   (ensemble/make identity leaf deep split)]
      (is (= 3 (tree-core/tree-depth ens))))))

(deftest test-ensemble-display
  (testing "pretty-printing the ensemble"
    (let [t1     (binary-tree/make-leaf 1)
          t2     (binary-tree/make-leaf 2)
          ens    (ensemble/make identity t1 t2)
          output (with-out-str (tree-core/display-tree ens))
          lines  (str/split-lines output)]
      (is (= "Ensemble Tree:" (nth lines 0)))
      (is (= " Root:  Ensemble of 2 trees" (nth lines 1))))))
