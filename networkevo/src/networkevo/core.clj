(ns networkevo.core
  (:use [clojush.ns]
        [clojush translate]
        networkevo.nn-instructions))

(use-clojush)

;; Params that will be in atom eventually
(def num-inputs 4)
(def num-outputs 3)
(def initial-nn-info
  {:layers  {:I {:num-inputs num-inputs
                 :num-outputs num-inputs}
             :O {:num-inputs num-inputs
                 :num-outputs num-outputs}}
   :layer-connections [[:I  :O]]})

(def atom-generators
  (concat (list 
            'nn_connect_layers
            'nn_bud
            'nn_split
            'nn_loop
            'nn_reverse
            'nn_set_num_inputs_to_layer
            'nn_set_num_outputs_to_layer
            (fn [] (lrand-int 100)))
          ;(registered-for-stacks [:exec :integer :nn])
          ))

(def program (translate-plush-genome-to-push-program
               {:genome
               (random-plush-genome 20
                                    atom-generators
                                    @push-argmap)}
               @push-argmap))

(println program)


(def s (run-push program 
                 (push-item initial-nn-info 
                            :auxilary
                            (make-push-state))))

(state-pretty-print s)
(println)

(println (:auxilary s))