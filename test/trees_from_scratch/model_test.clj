(ns trees-from-scratch.model-test
  (:require [clojure.test :refer [deftest is testing]]
            [trees-from-scratch.loss :as loss]
            [trees-from-scratch.dataset :as ds]
            [trees-from-scratch.model :as model]
            [trees-from-scratch.tree :as tree]))

(defn approx=
  ([a b] (approx= a b 1e-5))
  ([a b tol]
   (< (Math/abs (double (- a b))) tol)))

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

(deftest test-majority-class
  (testing "finding majority class label in a target vector"
    (is (= "Yes" (model/majority-class ["Yes" "No" "Yes" "Yes" "No"])))
    (is (= "A"   (model/majority-class ["A" "A" "A"]))))

  (testing "handling single element and empty input"
    (is (= "X" (model/majority-class ["X"])))
    (is (nil?  (model/majority-class [])))))

(deftest test-mean-target
  (testing "calculating mean target value for regression vector"
    (is (approx= 20.0 (model/mean-target [10.0 20.0 30.0])))
    (is (approx= 5.5 (model/mean-target [5.5]))))

  (testing "empty target vector defaults to 0.0"
    (is (approx= 0.0 (model/mean-target [])))))

(deftest test-impurity-gain
  (testing "perfect classification split produces maximum gain"
    (let [parent ["Yes" "Yes" "No" "No"]
          left   ["Yes" "Yes"]
          right  ["No" "No"]
          gain   (model/impurity-gain parent left right loss/gini)]
      ;; Parent Gini = 0.5, Left = 0.0, Right = 0.0 -> Gain = 0.5
      (is (approx= 0.5 gain))))

  (testing "useless split produces zero gain"
    (let [parent ["Yes" "No" "Yes" "No"]
          left   ["Yes" "No"]
          right  ["Yes" "No"]
          gain   (model/impurity-gain parent left right loss/gini)]
      (is (approx= 0.0 gain))))

  (testing "regression impurity gain using mean-squared-deviation"
    (let [parent [10.0 10.0 100.0 100.0]
          left   [10.0 10.0]
          right  [100.0 100.0]
          gain   (model/impurity-gain parent left right loss/mean-squared-deviation)]
      (is (> gain 0.0)))))

(deftest test-best-split
  (testing "finding optimal continuous split on dataset"
    (let [split (model/best-split classification-dataset :label {:impurity-fn loss/gini})]
      (is (some? split))
      (is (= :age (:feature split)))
      (is (> (:gain split) 0.0))
      (is (map? (:left split)))
      (is (map? (:right split)))
      (is (= 2 (ds/row-count (:left split))))
      (is (= 3 (ds/row-count (:right split))))))

  (testing "finding optimal split among categorical features"
    (let [split (model/best-split categorical-dataset :play {:features [:outlook :temperature]})]
      (is (some? split))
      (is (contains? #{:outlook :temperature} (:feature split)))
      (is (> (:gain split) 0.0))))

  (testing "returns nil or zero-gain split on pure dataset"
    (let [pure-dataset {:columns {:x [1 2] :label ["A" "A"]}
                        :types {:x :continuous :label :categorical}}
          split        (model/best-split pure-dataset :label {})]
      (is (or (nil? split) (<= (:gain split) 0.0))))))

(deftest test-cart-stop-tree?
  (testing "stopping due to reaching max depth"
    (is (true? (model/cart-early-exit? {:max-depth 5 :depth 5} classification-dataset :label)))
    (is (false? (model/cart-early-exit? {:max-depth 5 :depth 2} classification-dataset :label))))

  (testing "stopping due to minimum samples split threshold"
    (let [small-dataset {:columns {:x [1] :label ["A"]} :types {:x :continuous :label :categorical}}]
      (is (true? (model/cart-early-exit? {:min-samples-split 2 :depth 1} small-dataset :label))))
    (is (false? (model/cart-early-exit? {:min-samples-split 2 :depth 1} classification-dataset :label))))

  (testing "stopping when dataset target column is pure"
    (let [pure-data {:columns {:x [1 2] :label ["Yes" "Yes"]} :types {:x :continuous :label :categorical}}]
      (is (true? (model/cart-early-exit? {:depth 0} pure-data :label))))))

(deftest test-cart-late-exit?
  (testing "returns true if gain is less than or equal to min-gain"
    (let [split-record {:feature :x :gain 0.05}
          options      {:min-gain 0.1}]
      (is (true? (model/cart-late-exit? options split-record))))
    (let [split-record {:feature :x :gain 0.1}
          options      {:min-gain 0.1}]
      (is (true? (model/cart-late-exit? options split-record)))))

  (testing "returns false if gain is greater than min-gain"
    (let [split-record {:feature :x :gain 0.15}
          options      {:min-gain 0.1}]
      (is (false? (model/cart-late-exit? options split-record))))))

(deftest test-train-classification
  (testing "training a classification tree on composite dataset"
    (let [dt-model (model/train classification-dataset :label {:max-depth 3})]
      (is (map? dt-model))
      ;; Check root node is a split
      (is (= :split (:type dt-model)))
      ;; Verify predictions on single row maps
      (is (= "No"  (tree/predict dt-model {:age 15 :income 20})))
      (is (= "No"  (tree/predict dt-model {:age 22 :income 25})))
      (is (= "Yes" (tree/predict dt-model {:age 35 :income 70})))
      (is (= "Yes" (tree/predict dt-model {:age 50 :income 85}))))))

(deftest test-train-regression
  (testing "training a regression tree on continuous target data"
    (let [dt-model (model/train regression-dataset :salary {:task-type :regression
                                                            :impurity-fn loss/mean-squared-deviation
                                                            :max-depth 3})]
      (is (map? dt-model))
      (let [pred1 (tree/predict dt-model {:experience 1 :education "BSc"})
            pred2 (tree/predict dt-model {:experience 9 :education "PhD"})]
        (is (number? pred1))
        (is (number? pred2))
        ;; Low experience should yield lower salary prediction than high experience
        (is (< pred1 pred2))))))

(deftest test-train-with-depth-limit
  (testing "max-depth parameter constrains tree height"
    (let [shallow-tree (model/train categorical-dataset :play {:max-depth 1})]
      (is (<= (tree/tree-depth shallow-tree) 2)))))

(deftest test-predict-all
  (testing "predict-all returns vector of predictions"
    (let [dt-model (model/train classification-dataset :label {:max-depth 3})
          preds    (model/predict-all dt-model classification-dataset)]
      (is (= (ds/row-count classification-dataset) (count preds)))
      (is (= ["No" "No" "Yes" "Yes" "Yes"] preds)))))

(deftest test-evaluate
  (testing "evaluating classification accuracy"
    (let [dt-model (model/train classification-dataset :label {:max-depth 3})
          acc      (model/evaluate dt-model classification-dataset :label :accuracy)]
      (is (approx= 1.0 acc))))

  (testing "evaluating regression mean squared error (MSE)"
    (let [dt-model (model/train regression-dataset :salary {:task-type :regression
                                                            :max-depth 5})
          mse      (model/evaluate dt-model regression-dataset :salary :mse)]
      (is (number? mse))
      (is (>= mse 0.0)))))
