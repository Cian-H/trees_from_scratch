(ns trees-from-scratch.maths
  "Mathematical utility functions used throughout the library.")

(def ^:const ln2 (Math/log 2.0))

(defn logn
  "Computes the logarithm of x to the base n."
  [x n]
  (/ (Math/log x) (Math/log n)))

(defn log2
  "Computes the base-2 logarithm of x."
  [x]
  (/ (Math/log x) ln2))

(defn mean
  "Calculates the arithmetic mean of a collection of numbers."
  [coll]
  (let [n (count coll)]
    (if (zero? n)
      0.0
      (/ (double (reduce + coll)) n))))

(defn median
  "Calculates the median of a collection of numbers."
  [coll]
  (let [n (count coll)]
    (if (zero? n)
      0.0
      (let [sorted (vec (sort coll))
            half   (quot n 2)]
        (if (odd? n)
          (nth sorted half)
          (/ (+ (nth sorted (dec half))
                (nth sorted half))
             2.0))))))

(defn matrix-sub [x y]
  (cond
    (and (number? x) (number? y)) (- x y)

    (and (coll? x) (coll? y))
    (if (= (count x) (count y))
      (map matrix-sub x y)
      (throw (ex-info "Collections must have same dimensions"
                      {:reason :dimension-mismatch :inputs [(count x) (count y)]})))

    :else (throw (ex-info "Arguments must both be collections or both be numbers"
                          {:reason :type-mismatch :inputs [x y]}))))
