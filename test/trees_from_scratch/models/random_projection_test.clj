(ns trees-from-scratch.models.random-projection-test
  (:require [clojure.test :refer [deftest is testing]]
            [trees-from-scratch.data.dataset :as ds]
            [trees-from-scratch.models.random-projection :as rp]
            [trees-from-scratch.metrics.evaluation :as evaluation]))

(def regression-dataset
  {:columns {:experience [1 2 5 8 9]
             :education  ["BSc" "BSc" "MSc" "MSc" "PhD"]
             :salary     [45.0 50.0 75.0 110.0 125.0]}
   :types   {:experience :continuous
             :education  :categorical
             :salary     :continuous}})

(deftest test-train-regression
  (testing "training a random projection tree on regression dataset"
    ;; Note: Categorical features are ignored, so it should only train on :experience
    (let [rp-model (rp/train regression-dataset :salary {:task-type :regression})]
      (is (some? rp-model))
      (let [preds (evaluation/predict-all rp-model regression-dataset)]
        (is (= (ds/row-count regression-dataset) (count preds)))))))

(deftest test-train-classification-throws
  (testing "training random projection tree on classification throws exception"
    (is (thrown? clojure.lang.ExceptionInfo (rp/train regression-dataset :salary {:task-type :classification})))))
