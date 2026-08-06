(ns trees-from-scratch.models.gradient-boosted-test
  (:require [clojure.test :refer [deftest is testing]]
            [trees-from-scratch.data.dataset :as ds]
            [trees-from-scratch.models.gradient-boosted :as gbdt]
            [trees-from-scratch.metrics.evaluation :as evaluation]))

(def regression-dataset
  {:columns {:experience [1 2 5 8 9]
             :education  ["BSc" "BSc" "MSc" "MSc" "PhD"]
             :salary     [45.0 50.0 75.0 110.0 125.0]}
   :types   {:experience :continuous
             :education  :categorical
             :salary     :continuous}})

(deftest test-train-regression
  (testing "training a gradient boosted tree on regression dataset"
    (let [gbdt-model (gbdt/train regression-dataset :salary {:task-type :regression :learning-rate 0.1 :max-trees 5})]
      (is (some? gbdt-model))
      (let [preds (evaluation/predict-all gbdt-model regression-dataset)]
        (is (= (ds/row-count regression-dataset) (count preds)))))))

(deftest test-train-classification-throws
  (testing "training GBDT on classification throws exception"
    (is (thrown? clojure.lang.ExceptionInfo (gbdt/train regression-dataset :salary {:task-type :classification})))))
