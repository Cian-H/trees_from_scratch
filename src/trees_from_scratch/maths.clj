(ns trees-from-scratch.maths)

(def ^:const ln2 (Math/log 2.0))

(defn logn [x n]
  (/ (Math/log x) (Math/log n)))

(defn log2 [x]
  (/ (Math/log x) ln2))

(defn mean [coll]
  (let [n (count coll)]
    (if (zero? n)
      0.0
      (/ (double (reduce + coll)) n))))

(defn median [coll]
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
