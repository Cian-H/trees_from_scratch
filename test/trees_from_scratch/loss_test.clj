(ns trees-from-scratch.loss-test
  (:require [clojure.test :refer [deftest is testing]]
            [trees-from-scratch.loss :refer [gini
                                             entropy
                                             misclassification-rate
                                             mean-squared-deviation
                                             mean-absolute-deviation
                                             poisson-deviance]]))

(defn approx=
  ([expected actual]
   (approx= expected actual 1e-5))
  ([expected actual tolerance]
   (<= (Math/abs (- (double expected) (double actual))) tolerance)))

(deftest test-gini
  (testing "Perfect purity (one class)"
    (is (== 0.0 (gini [1 1 1]))))

  (testing "50/50 split"
    (is (approx= 0.5 (gini [0 1 0 1]))))

  (testing "Even split across 3 classes"
    (is (approx= 0.6666666 (gini [0 1 2]))))

  (testing "Empty node"
    (is (== 0.0 (gini [])))))

(deftest test-entropy
  (testing "Perfect purity"
    (is (== 0.0 (entropy [0 0 0]))))

  (testing "50/50 split"
    (is (approx= 1.0 (entropy [1 0 1 0]))))

  (testing "Empty node"
    (is (== 0.0 (entropy [])))))

(deftest test-misclassification-rate
  (testing "Perfect purity"
    (is (== 0.0 (misclassification-rate [2 2]))))

  (testing "75% majority class"
    (is (approx= 0.25 (misclassification-rate [0 0 0 1]))))

  (testing "50/50 split"
    (is (approx= 0.5 (misclassification-rate [0 1]))))

  (testing "Empty node"
    (is (== 0.0 (misclassification-rate [])))))

(deftest test-mean-squared-deviation
  (testing "Zero variance (all values identical)"
    (is (== 0.0 (mean-squared-deviation [5.0 5.0 5.0]))))

  (testing "Known variance"
    (is (approx= 1.0 (mean-squared-deviation [2.0 4.0]))))

  (testing "Empty node"
    (is (== 0.0 (mean-squared-deviation [])))))

(deftest test-mean-absolute-deviation
  (testing "Zero error"
    (is (== 0.0 (mean-absolute-deviation [10.0 10.0]))))

  (testing "Known MAE"
    (is (approx= 1.3333333 (mean-absolute-deviation [1.0 3.0 5.0]))))

  (testing "Empty node"
    (is (== 0.0 (mean-absolute-deviation [])))))

(deftest test-poisson-deviance
  (testing "All zeros (perfect purity for counts)"
    (is (== 0.0 (poisson-deviance [0 0]))))

  (testing "Constant non-zero counts"
    (is (== 0.0 (poisson-deviance [2 2 2]))))

  (testing "Known deviance for [1, 3]"
    (let [expected-deviance (* 2.0 (+ (* 1.0 (Math/log 0.5))
                                      1.0
                                      (* 3.0 (Math/log 1.5)) -1.0))]
      (is (approx= expected-deviance (poisson-deviance [1 3])))))

  (testing "Handle arrays with zeros properly (should not throw log(0) errors)"
    (let [expected-with-zero (* 2.0 (+ 0.0 (* 2.0 (Math/log 2.0)) -1.0))]
      (is (approx= expected-with-zero (poisson-deviance [0 2])))))

  (testing "Empty node"
    (is (== 0.0 (poisson-deviance [])))))
