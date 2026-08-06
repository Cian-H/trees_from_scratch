(ns trees-from-scratch.models.cart-test
  (:require [clojure.test :refer [deftest is testing]]
            [trees-from-scratch.metrics.loss :as loss]
            [trees-from-scratch.data.dataset :as ds]
            [trees-from-scratch.models.cart :as cart]
            [trees-from-scratch.trees.core :as tree-core]
            [trees-from-scratch.metrics.evaluation :as evaluation]
            [trees-from-scratch.utils.core-test :refer [approx=]]))

(def classification-dataset
  {:columns {:age    [15 22 35 45 50]
             :income [20 25 70 90 85]
             :label  ["No" "No" "Yes" "Yes" "Yes"]}
   :types   {:age    :continuous
             :income :continuous
             :label  :categorical}})

(def categorical-dataset
  {:columns {:outlook     ["sunny" "sunny" "overcast" "rain" "rain" "rain" "overcast" "sunny" "sunny" "rain"]
             :temperature ["hot" "hot" "hot" "mild" "cool" "cool" "cool" "mild" "cool" "mild"]
             :play        ["no" "no" "yes" "yes" "yes" "no" "yes" "no" "yes" "yes"]}
   :types   {:outlook     :categorical
             :temperature :categorical
             :play        :categorical}})

(def regression-dataset
  {:columns {:experience [1 2 5 8 9]
             :education  ["BSc" "BSc" "MSc" "MSc" "PhD"]
             :salary     [45.0 50.0 75.0 110.0 125.0]}
   :types   {:experience :continuous
             :education  :categorical
             :salary     :continuous}})

(deftest test-loss-reduction
  (testing "perfect classification split produces maximum reduction"
    (let [parent ["Yes" "Yes" "No" "No"]
          left   ["Yes" "Yes"]
          right  ["No" "No"]
          reduction (evaluation/loss-reduction parent left right loss/gini)]
      (is (approx= 0.5 reduction))))

  (testing "useless split produces zero reduction"
    (let [parent ["Yes" "No" "Yes" "No"]
          left   ["Yes" "No"]
          right  ["Yes" "No"]
          reduction (evaluation/loss-reduction parent left right loss/gini)]
      (is (approx= 0.0 reduction))))

  (testing "regression loss reduction using mean-squared-deviation"
    (let [parent [10.0 10.0 100.0 100.0]
          left   [10.0 10.0]
          right  [100.0 100.0]
          reduction (evaluation/loss-reduction parent left right loss/mean-squared-deviation)]
      (is (> reduction 0.0)))))

(deftest test-best-split
  (testing "finding optimal continuous split on dataset"
    (let [split (cart/best-split classification-dataset :label {:loss-fn loss/gini})]
      (is (some? split))
      (is (= :age (:feature split)))
      (is (> (:loss-reduction split) 0.0))
      (is (map? (:left split)))
      (is (map? (:right split)))
      (is (= 2 (ds/row-count (:left split))))
      (is (= 3 (ds/row-count (:right split))))))

  (testing "finding optimal split among categorical features"
    (let [split (cart/best-split categorical-dataset :play {:features [:outlook :temperature]})]
      (is (some? split))
      (is (contains? #{:outlook :temperature} (:feature split)))
      (is (> (:loss-reduction split) 0.0))))

  (testing "returns nil or zero-reduction split on pure dataset"
    (let [pure-dataset {:columns {:x [1 2] :label ["A" "A"]}
                        :types {:x :continuous :label :categorical}}
          split        (cart/best-split pure-dataset :label {})]
      (is (or (nil? split) (<= (:loss-reduction split) 0.0))))))

(deftest test-train-classification
  (testing "training a classification tree on composite dataset"
    (let [dt-model (cart/train classification-dataset :label {:max-depth 3})]
      (is (map? dt-model))
      ;; Check root node is a split
      (is (instance? trees_from_scratch.trees.binary.BinarySplit dt-model))
      ;; Verify predictions on single row maps
      (is (= "No"  (tree-core/predict dt-model {:age 15 :income 20})))
      (is (= "No"  (tree-core/predict dt-model {:age 22 :income 25})))
      (is (= "Yes" (tree-core/predict dt-model {:age 35 :income 70})))
      (is (= "Yes" (tree-core/predict dt-model {:age 50 :income 85}))))))

(deftest test-train-regression
  (testing "training a regression tree on continuous target data"
    (let [dt-model (cart/train regression-dataset :salary {:task-type :regression
                                                           :loss-fn loss/mean-squared-deviation
                                                           :max-depth 3})]
      (is (map? dt-model))
      (let [pred1 (tree-core/predict dt-model {:experience 1 :education "BSc"})
            pred2 (tree-core/predict dt-model {:experience 9 :education "PhD"})]
        (is (number? pred1))
        (is (number? pred2))
        ;; Low experience should yield lower salary prediction than high experience
        (is (< pred1 pred2))))))

(deftest test-train-with-depth-limit
  (testing "max-depth parameter constrains tree height"
    (let [shallow-tree (cart/train categorical-dataset :play {:max-depth 1})]
      (is (<= (tree-core/tree-depth shallow-tree) 2)))))

(deftest test-predict-all
  (testing "predict-all returns vector of predictions"
    (let [dt-model (cart/train classification-dataset :label {:max-depth 3})
          preds    (evaluation/predict-all dt-model classification-dataset)]
      (is (= (ds/row-count classification-dataset) (count preds)))
      (is (= ["No" "No" "Yes" "Yes" "Yes"] preds)))))

(deftest test-evaluate
  (testing "evaluating classification accuracy"
    (let [dt-model (cart/train classification-dataset :label {:max-depth 3})
          acc      (evaluation/evaluate dt-model classification-dataset :label :accuracy)]
      (is (approx= 1.0 acc))))

  (testing "evaluating regression mean squared error (MSE)"
    (let [dt-model (cart/train regression-dataset :salary {:task-type :regression
                                                           :max-depth 5})
          mse      (evaluation/evaluate dt-model regression-dataset :salary :mse)]
      (is (number? mse))
      (is (>= mse 0.0)))))
