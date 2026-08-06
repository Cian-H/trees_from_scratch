(ns trees-from-scratch.models.random-forest-test
  (:require [clojure.test :refer [deftest is testing]]
            [trees-from-scratch.data.dataset :as ds]
            [trees-from-scratch.models.random-forest :as rf]
            [trees-from-scratch.metrics.evaluation :as evaluation]))

(def classification-dataset
  {:columns {:age    [15 22 35 45 50]
             :income [20 25 70 90 85]
             :label  ["No" "No" "Yes" "Yes" "Yes"]}
   :types   {:age    :continuous
             :income :continuous
             :label  :categorical}})

(def regression-dataset
  {:columns {:experience [1 2 5 8 9]
             :education  ["BSc" "BSc" "MSc" "MSc" "PhD"]
             :salary     [45.0 50.0 75.0 110.0 125.0]}
   :types   {:experience :continuous
             :education  :categorical
             :salary     :continuous}})

(deftest test-train-classification
  (testing "training a random forest on classification dataset"
    (let [rf-model (rf/train classification-dataset :label {:task-type :classification :window-size 2 :min-delta 0.0})]
      (is (some? rf-model))
      (let [preds (evaluation/predict-all rf-model classification-dataset)]
        (is (= (ds/row-count classification-dataset) (count preds)))))))

(deftest test-train-regression
  (testing "training a random forest on regression dataset"
    (let [rf-model (rf/train regression-dataset :salary {:task-type :regression :window-size 2 :min-delta 0.0})]
      (is (some? rf-model))
      (let [preds (evaluation/predict-all rf-model regression-dataset)]
        (is (= (ds/row-count regression-dataset) (count preds)))))))
