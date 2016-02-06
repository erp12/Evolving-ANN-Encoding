(ns networkevo.nn-instructions
  (:use [clojush pushstate globals]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Utility Functions

(defn insert 
  "https://groups.google.com/forum/#!msg/clojure/zjZDSziZVQk/58xCUSZYYPwJ"
  [vec pos item] 
  (apply merge (subvec vec 0 pos) item (subvec vec pos)))

(defn insert-hidden-layer
  "Returns network-info with new hidden layer."
  [network-info num-inputs num-outputs]
  (let [layers (:layers network-info)
        hidden-layers (dissoc (dissoc layers
                                      :I)
                              :O)
        next-hidden-id (symbol (str ":H" 
                                    (inc (count hidden-layers))))]
    [(assoc-in network-info
               [:layers next-hidden-id]
               {:num-inputs num-inputs
                :num-outputs num-outputs})
     next-hidden-id]))

(defn new-layer-connection
  "Returns network-info with a new layer connection."
  [network-info [from-id to-id]]
  (let [layer-conns (:layer-connections network-info)
        output-connections (filter #(= :O (nth % 1)) 
                                   layer-conns)
        non-output-connections (remove  #(= :O (nth % 1))
                                        layer-conns)
        new-layer-conns (concat (conj (vec non-output-connections)
                                      [from-id to-id])
                                output-connections)]
    (if (some #(= [from-id to-id] %) layer-conns)
      network-info
      (assoc network-info :layer-connections new-layer-conns))))

(defn remove-layer-connection
  "Returns network-info with layer connection removed."
  [network-info [from-id to-id]]
  (let [layer-conns (:layer-connections network-info)
        new-layer-conns (remove #(= [from-id to-id] %)
                                layer-conns)]
    (assoc network-info :layer-connections new-layer-conns)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Instructions


(define-registered
  nn_connect_layers
  ^{:stack-types [:auxilary :integer :nn]}
(fn [state]
  (if (and (not (empty? (rest (:integer state))))
           (not (empty? (:auxilary  state))))
    (let [nn-info (top-item :auxilary state)
          to-index (top-item :integer state)
          from-index (top-item :integer (pop-item :integer state))
          from-id (nth (keys (:layers nn-info))
                       (mod from-index
                            (count (keys (:layers nn-info)))))
          to-id (nth (keys (:layers nn-info))
                     (mod to-index
                          (count (keys (:layers nn-info)))))
          new-nn-info (new-layer-connection nn-info [from-id to-id])]
      (->> 
        (pop-item :integer state)
        (pop-item :integer)
        (pop-item :auxilary)
        (push-item new-nn-info :auxilary)))
    state)))

(define-registered
  nn_bud
  ^{:stack-types [:auxilary :integer :nn]}
(fn [state]
  (if (and (not (empty? (:integer  state)))
           (not (empty? (:auxilary  state))))
    (let [nn-info (top-item :auxilary state)
          edge-to-bud (nth (:layer-connections nn-info)
                           (mod (stack-ref :integer 0 state)
                                 (count (:layer-connections nn-info))))
          num-input-and-outputs (:num-outputs (get (:layers nn-info) 
                                                   (second edge-to-bud)))
          
          temp (insert-hidden-layer nn-info 
                                    num-input-and-outputs
                                    num-input-and-outputs)
          new-nn-info (first temp)
          new-hidden-id (second temp)
          
          new-new-nn-info (new-layer-connection new-nn-info [(second edge-to-bud)
                                                             new-hidden-id])]
      (->> 
        (pop-item :integer state)
        (pop-item :auxilary)
        (push-item new-new-nn-info :auxilary)))
    state)))

(define-registered
  nn_split
  ^{:stack-types [:auxilary :integer :nn]}
(fn [state]
  (if (and (not (empty? (:integer  state)))
           (not (empty? (:auxilary  state))))
    (let [nn-info (top-item :auxilary state)
          edge-to-split (nth (:layer-connections nn-info)
                             (mod (stack-ref :integer 0 state)
                                 (count (:layer-connections nn-info))))
          num-inputs (:num-outputs (get (:layers nn-info) 
                                        (first edge-to-split)))
          num-outputs (:num-inputs (get (:layers nn-info) 
                                        (first edge-to-split)))
          
          temp (insert-hidden-layer (remove-layer-connection nn-info 
                                                             edge-to-split)
                                    num-inputs 
                                    num-outputs)
          nn-info (first temp)
          new-hidden-id (second temp)
          
          nn-info (new-layer-connection nn-info [(first edge-to-split) new-hidden-id])
          nn-info (new-layer-connection nn-info [new-hidden-id (second edge-to-split)])]
      (->> 
        (pop-item :integer state)
        (pop-item :auxilary)
        (push-item nn-info :auxilary)))
    state)))

(define-registered
  nn_loop
  ^{:stack-types [:auxilary :integer :nn]}
(fn [state]
  (if (and (not (empty? (:integer  state)))
           (not (empty? (:auxilary  state))))
    (let [nn-info (top-item :auxilary state)
          edge-to-loop (nth (:layer-connections nn-info)
                            (mod (stack-ref :integer 0 state)
                                 (count (:layer-connections nn-info)))) 
          nn-info (new-layer-connection nn-info [(second edge-to-loop)
                                                 (first edge-to-loop)])]
      (->> 
        (pop-item :integer state)
        (pop-item :auxilary)
        (push-item nn-info :auxilary)))
    state)))

(define-registered
  nn_reverse
  ^{:stack-types [:auxilary :integer :nn]}
(fn [state]
  (if (and (not (empty? (:integer  state)))
           (not (empty? (:auxilary  state))))
    (let [nn-info (top-item :auxilary state)
          edge-to-reverse (nth (:layer-connections nn-info)
                               (mod (stack-ref :integer 0 state)
                                 (count (:layer-connections nn-info))))
          nn-without-edge (remove-layer-connection nn-info edge-to-reverse)
          nn-info (new-layer-connection nn-without-edge [(second edge-to-reverse)
                                                         (first edge-to-reverse)])]
      (->> 
        (pop-item :integer state)
        (pop-item :auxilary)
        (push-item nn-info :auxilary)))
    state)))

(define-registered
  nn_set_num_inputs_to_layer
  ^{:stack-types [:auxilary :integer :nn]}
(fn [state]
  (if (and (not (empty? (rest (:integer  state))))
           (not (empty? (:auxilary  state))))
    (let [nn-info (top-item :auxilary state)
          layer-ids (keys (:layers nn-info))
          layer-id (nth layer-ids
                        (mod (stack-ref :integer 0 state)
                             (count layer-ids)))
          new-num-inputs (stack-ref :integer 1 state)]
      (if (= layer-id :I)
        state
        (let [new-nn-info (assoc-in nn-info
                                    [:layers layer-id :num-inputs]
                                    new-num-inputs)
              pre-layers (map first
                              (filter #(= (second %) layer-id)
                                      (:layer-connections new-nn-info)))
              changed-layers (map (fn [x] 
                                    {x (assoc (x (:layers new-nn-info))
                                              :num-outputs
                                              new-num-inputs)}) 
                                  pre-layers)
              new-nn-info (assoc new-nn-info
                                 :layers
                                 (merge (:layers new-nn-info)
                                        (apply merge changed-layers)))]
          (->> 
            (pop-item :integer state)
            (pop-item :integer)
            (pop-item :auxilary)
            (push-item new-nn-info :auxilary)))))
    state)))

(define-registered
  nn_set_num_outputs_to_layer
  ^{:stack-types [:auxilary :integer :nn]}
(fn [state]
  (if (and (not (empty? (rest (:integer  state))))
           (not (empty? (:auxilary  state))))
    (let [nn-info (top-item :auxilary state)
          layer-ids (keys (:layers nn-info))
          layer-id (nth layer-ids
                        (mod (stack-ref :integer 0 state)
                             (count layer-ids)))
          new-num-outputs (stack-ref :integer 1 state)]
      (if (= layer-id :O)
        state
        (let [new-nn-info (assoc-in nn-info
                                    [:layers layer-id :num-outputs]
                                    new-num-outputs)
              post-layers (map second
                               (filter #(= (first %) layer-id)
                                       (:layer-connections new-nn-info)))
              changed-layers (map (fn [x] 
                                    {x (assoc (x (:layers new-nn-info))
                                              :num-inputs
                                              new-num-outputs)}) 
                                  post-layers)
              new-nn-info (assoc new-nn-info
                                 :layers
                                 (merge (:layers new-nn-info)
                                        (apply merge changed-layers)))]
          (->> 
            (pop-item :integer state)
            (pop-item :integer)
            (pop-item :auxilary)
            (push-item new-nn-info :auxilary)))))
    state)))