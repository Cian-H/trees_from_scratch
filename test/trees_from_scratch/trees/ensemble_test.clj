(ns trees-from-scratch.trees.ensemble-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [trees-from-scratch.trees.core :as tree-core]
            [trees-from-scratch.trees.ensemble :as ensemble]
            [trees-from-scratch.trees.binary :as binary-tree])
  (:import [trees_from_scratch.trees.ensemble.parallel ParallelEnsemble]
           [trees_from_scratch.trees.ensemble.cascade CascadeEnsemble]))

(deftest test-ensemble-creation
  (testing "make creates a ParallelEnsemble record by default"
    (let [agg   identity
          t1    (binary-tree/make-leaf 1)
          ens   (ensemble/make agg t1)]
      (is (instance? ParallelEnsemble ens))
      (is (= agg (:agg ens)))
      (is (= [t1] (:trees ens)))))

  (testing "make-parallel explicitly creates ParallelEnsemble"
    (let [agg identity
          t1  (binary-tree/make-leaf 1)
          ens (ensemble/make-parallel agg t1)]
      (is (instance? ParallelEnsemble ens))))

  (testing "make-cascade creates CascadeEnsemble"
    (let [agg identity
          t1  (binary-tree/make-leaf 1)
          ens (ensemble/make-cascade agg t1)]
      (is (instance? CascadeEnsemble ens)))))

(deftest test-parallel-ensemble-predict
  (testing "parallel ensemble predicts by aggregating sub-tree predictions"
    (let [t1    (binary-tree/make-leaf 10)
          t2    (binary-tree/make-leaf 20)
          t3    (binary-tree/make-leaf 30)
          agg   (fn [preds] (reduce + preds))
          ens   (ensemble/make-parallel agg t1 t2 t3)]
      (is (= 60 (tree-core/predict ens {:any "row"})))))

  (testing "parallel ensemble handles majority vote"
    (let [t1    (binary-tree/make-leaf "A")
          t2    (binary-tree/make-leaf "B")
          t3    (binary-tree/make-leaf "A")
          agg   (fn [preds] (->> preds frequencies (sort-by val >) ffirst))
          ens   (ensemble/make-parallel agg t1 t2 t3)]
      (is (= "A" (tree-core/predict ens {}))))))

(deftest test-cascade-ensemble-predict
  (testing "cascade ensemble evaluates trees sequentially, passing prior predictions"
    ;; Tree 1: predicts 10
    ;; Tree 2: split based on :prev-pred from Tree 1. If :prev-pred is 10, predict 100, else 0.
    (let [t1    (binary-tree/make-leaf 10)
          t2    (binary-tree/make-continuous-split :prev-pred 10
                                                   (binary-tree/make-leaf 100)
                                                   (binary-tree/make-leaf 0))
          agg   (fn [preds] (last preds))
          ens   (ensemble/make-cascade agg t1 t2)]
      (is (= 100 (tree-core/predict ens {})))))

  (testing "cascade ensemble aggregates all stage predictions if agg is provided"
    (let [t1    (binary-tree/make-leaf 5)
          t2    (binary-tree/make-leaf 15)
          agg   (fn [preds] (reduce + preds))
          ens   (ensemble/make-cascade agg t1 t2)]
      (is (= 20 (tree-core/predict ens {}))))))

(deftest test-ensemble-tree-depth
  (testing "depth of empty parallel ensemble is 0"
    (let [ens (ensemble/make-parallel identity)]
      (is (= 0 (tree-core/tree-depth ens)))))

  (testing "depth of parallel ensemble is the max depth of its sub-trees"
    (let [leaf  (binary-tree/make-leaf 1)                                   ;; depth 1
          split (binary-tree/make-continuous-split :x 10 leaf leaf)         ;; depth 2
          deep  (binary-tree/make-continuous-split :y 20 split split)       ;; depth 3
          ens   (ensemble/make-parallel identity leaf deep split)]
      (is (= 3 (tree-core/tree-depth ens)))))

  (testing "depth of cascade ensemble is the sum of depths of its sub-trees"
    (let [leaf  (binary-tree/make-leaf 1)                                   ;; depth 1
          split (binary-tree/make-continuous-split :x 10 leaf leaf)         ;; depth 2
          deep  (binary-tree/make-continuous-split :y 20 split split)       ;; depth 3
          ens   (ensemble/make-cascade identity leaf deep split)]
      (is (= 6 (tree-core/tree-depth ens))))))

(deftest test-ensemble-display
  (testing "pretty-printing parallel ensemble"
    (let [t1     (binary-tree/make-leaf 1)
          t2     (binary-tree/make-leaf 2)
          ens    (ensemble/make-parallel identity t1 t2)
          output (with-out-str (tree-core/display-tree ens))
          lines  (str/split-lines output)]
      (is (= "Parallel Ensemble Tree:" (nth lines 0)))
      (is (= " Root:  Parallel Ensemble of 2 trees" (nth lines 1)))))

  (testing "pretty-printing cascade ensemble"
    (let [t1     (binary-tree/make-leaf 1)
          t2     (binary-tree/make-leaf 2)
          ens    (ensemble/make-cascade identity t1 t2)
          output (with-out-str (tree-core/display-tree ens))
          lines  (str/split-lines output)]
      (is (= "Cascade Ensemble Tree:" (nth lines 0)))
      (is (= " Root:  Cascade Ensemble of 2 trees" (nth lines 1))))))
