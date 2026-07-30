(ns trees-from-scratch.trees.binary-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [trees-from-scratch.trees.binary :as binary-tree]))

(deftest test-node-creation
  (testing "creating a leaf node (the answer)"
    (let [leaf (binary-tree/make-leaf "Spam")]
      (is (= "Spam" (:prediction leaf)))
      (is (nil? (:predicate leaf)))
      (is (nil? (:left leaf)))
      (is (nil? (:right leaf)))))

  (testing "creating an internal node (the question)"
    (let [node (binary-tree/make-continuous-split :age 30 nil nil)]
      (is (fn? (:predicate node)))
      (is (nil? (:prediction node)))
      (is (nil? (:left node)))
      (is (nil? (:right node))))))

(deftest test-predict
  (testing "routing unseen data through a decision tree"
    ; Manually build a small tree to test the prediction routing logic.
    ; Rule: If :age <= 30, predict "Young", else predict "Old"
    (let [leaf-left  (binary-tree/make-leaf "Young")
          leaf-right (binary-tree/make-leaf "Old")
          dt-tree    (binary-tree/make-continuous-split :age 30 leaf-left leaf-right)]

      (testing "Data passing the rule (<= 30) goes to the left child"
        (is (= "Young" (binary-tree/predict dt-tree {:age 25 :income 50000})))
        (is (= "Young" (binary-tree/predict dt-tree {:age 30 :income 100000}))))

      (testing "Data failing the rule (> 30) goes to the right child"
        (is (= "Old" (binary-tree/predict dt-tree {:age 31 :income 40000})))
        (is (= "Old" (binary-tree/predict dt-tree {:age 80 :income 0}))))

      (testing "Edge Case: Tree is just a single leaf"
        (is (= "Young" (binary-tree/predict leaf-left {:age 99})))))))

(deftest test-tree-depth
  (testing "calculating the depth of a decision tree structure"
    (let [leaf       (binary-tree/make-leaf "A")
          node-1     (binary-tree/make-continuous-split :x 1 leaf leaf)
          root-node  (binary-tree/make-continuous-split :y 2 node-1 leaf)]
      (is (= 0 (binary-tree/tree-depth nil)))
      (is (= 1 (binary-tree/tree-depth leaf)))
      (is (= 2 (binary-tree/tree-depth node-1)))
      (is (= 3 (binary-tree/tree-depth root-node))))))

(deftest test-display
  (testing "pretty-printing the decision tree structure to the console"
    (let [leaf-left  (binary-tree/make-leaf "Young")
          leaf-right (binary-tree/make-leaf "Old")
          dt-tree    (binary-tree/make-continuous-split :age 30 leaf-left leaf-right)
          output     (with-out-str (binary-tree/display dt-tree))
          lines      (str/split-lines output)]
      (is (= "Decision Tree:" (nth lines 0)))
      (is (= " Root:  Is :age <= 30 ?" (nth lines 1)))
      (is (= "   ├── True:   Predict -> Young" (nth lines 2)))
      (is (= "   └── False:  Predict -> Old" (nth lines 3))))))

(deftest test-categorical-split
  (testing "routing data using categorical split"
    (let [leaf-red   (binary-tree/make-leaf "Apple")
          leaf-other (binary-tree/make-leaf "Unknown")
          dt-tree    (binary-tree/make-categorical-split :color "red" leaf-red leaf-other)]
      (is (= "Apple" (binary-tree/predict dt-tree {:color "red"})))
      (is (= "Unknown" (binary-tree/predict dt-tree {:color "blue"})))
      (is (= "Unknown" (binary-tree/predict dt-tree {}))))))

(deftest test-predict-edge-cases
  (testing "predicting on nil tree returns nil"
    (is (nil? (binary-tree/predict nil {:age 25}))))

  (testing "predicting when feature key is missing routes to right branch"
    (let [leaf-left  (binary-tree/make-leaf "Young")
          leaf-right (binary-tree/make-leaf "Unknown")
          dt-tree    (binary-tree/make-continuous-split :age 30 leaf-left leaf-right)]
      (is (= "Unknown" (binary-tree/predict dt-tree {:income 50000}))))))
