(ns trees-from-scratch.models.oblique-test
  (:require [clojure.test :refer [deftest is testing]]
            [trees-from-scratch.data.dataset :as ds]
            [trees-from-scratch.models.oblique :as obl]
            [trees-from-scratch.metrics.evaluation :as evaluation]))

(def regression-dataset
  {:columns {:experience [1 2 5 8 9]
             :education  ["BSc" "BSc" "MSc" "MSc" "PhD"]
             :salary     [45.0 50.0 75.0 110.0 125.0]}
   :types   {:experience :continuous
             :education  :categorical
             :salary     :continuous}})

(deftest test-train-regression
  (testing "training an oblique tree on regression dataset"
    (let [obl-model (obl/train regression-dataset :salary {:task-type :regression})]
      (is (some? obl-model))
      (let [preds (evaluation/predict-all obl-model regression-dataset)]
        (is (= (ds/row-count regression-dataset) (count preds)))))))

(deftest test-train-classification-throws
  (testing "training oblique tree on classification throws exception"
    (is (thrown? clojure.lang.ExceptionInfo (obl/train regression-dataset :salary {:task-type :classification})))))
