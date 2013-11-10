
(defun test (out results &optional (i 0) (diffs nil))
   (if results
       (test out
             (cdr results)
             (1+ i)
             (cons (list (mem-aref out i) (car results)) diffs))
       diffs))

(defun calculate-neuron (net-count count mapping inp off wei additionals)
  (labels ((map-inputs (id m &optional (i 0))
              (when m
                (destructuring-bind (siz name start end) (car m)
                  (append (loop for n from start below end
                                    collect (mem-aref (if name (nth i additionals) inp)
                                                      (+ (* siz id) n)))
                              (map-inputs id (cdr m) (if name (+ 1 i) i))))))
           (summarize (inputs i sum)
              (if inputs
                (summarize (cdr inputs) (1+ i) (+ sum (* (car inputs) (mem-aref wei i))))
                sum)))
    (loop for id from 0 below net-count append
      (let ((inputs (map-inputs id mapping)))
        (loop for n from 0 below count collect
          (tanh (+ (mem-aref off (+ (* id count) n))
                   (summarize inputs (* (+ (* id count) n) (list-length inputs)) 0))))))))

(defun validate-neuron (net-count count mapping action inp out off wei &rest additionals)
  (let ((cpu-results (calculate-neuron net-count count mapping inp off wei additionals)))
      (funcall action)
      (memcpy-device-to-host out)
      (nreverse (test out cpu-results))))

(defun validate-memory (net-count count mapping action inp out off wei mem dat &rest additionals)
  (let* ((gate-results (calculate-neuron net-count (* 4 count) mapping inp off wei additionals))
         (cpu-results
           (loop for n from 0 below (* net-count 4 count) by 4 collect
             (destructuring-bind (input store give keep last)
               (list (nth n gate-results)
                     (round (/ (+ 1.0 (nth (+ 1 n) gate-results)) 2.0))
                     (round (/ (+ 1.0 (nth (+ 2 n) gate-results)) 2.0))
                     (round (/ (+ 1.0 (nth (+ 3 n) gate-results)) 2.0))
                     (mem-aref dat (/ n 4)))
                (tanh (* give (+ (* keep last) (* store input))))))))
    (funcall action)
    (memcpy-device-to-host out)
    (memcpy-device-to-host mem)
    (print 'memory-gates)
    (print (nreverse (test mem gate-results)))
    (print 'memory-outputs)
    (print (nreverse (test out cpu-results)))))


(defun cpu-layer-action (all-layers count)
  #'(lambda (layer)
     (destructuring-bind (name inputs outputs) (subseq layer 0 3)
       (let* ((input-vars (get-inputs inputs all-layers))
              (mapping (mapcar #'(lambda(i)
                                   (cons (or (third (find (first i) all-layers :key #'first))
                                             (count-inputs inputs)) i))
                               inputs)))
          (if (mem-p name)
            `(validate-memory ,(min 2 count) ,outputs ',mapping #',@(names layer 'act)
                ,@(names layer 'inp 'out 'off 'wei 'mem 'dat) ,@input-vars)
            `(print (validate-neuron ,(min 2 count) ,outputs ',mapping #',@(names layer 'act)
                ,@(names layer 'inp 'out 'off 'wei) ,@input-vars)))))))

(defun create-validator (layers count)
  (mapcar (cpu-layer-action layers count) layers))




; (defun main ()
;   (with-cuda-context (0)
;     (with-neural-networks rat
;                           2
;                           ;name   inputs                outputs
;                           ((A ((nil 0 3) (mem-b 0 3)) 3)
;                            (MEM-B ((a 0 3)) 3))
;       ; (rat-dissect 0 rat-reg-0)
;       ; (format t "~%~%################## test execution ####################~%~%")
;       ; (validate)
;       ; (format t "~%~%################## test transport ####################~%~%")
;       ; (rat-stitch 0 rat-reg-0)
;       ; (rat-stitch 1 rat-reg-0)
;       ; (validate)
;       ; (format t "~%################## test crossover ####################~%~%")
;       ; (rat-dissect 0 rat-reg-0)
;       ; (rat-dissect 1 rat-reg-1)
;       ; (memcpy-device-to-host rat-reg-0 rat-reg-1)
;       ; (rat-crossover rat-reg-2 rat-reg-0 rat-reg-1)
;       ; (print (loop for i from 0 to 98 collect
;       ;   (format nil "~a parent-a ~a, parent-b ~a, result ~a" i
;       ;           (mem-aref rat-reg-0 i)
;       ;           (mem-aref rat-reg-1 i)
;       ;           (mem-aref rat-reg-2 i))))
;       (format t "~%~%################## test mutation ####################~%~%")
;       (rat-dissect 0 rat-reg-0)
;       (rat-dissect 1 rat-reg-1)
;       (memcpy-device-to-host rat-reg-0 rat-reg-1)
;       (rat-mutation rat-reg-2 rat-reg-0)
;       (print (loop for i from 0 to 98 collect
;         (format nil "~a parent-a ~a, result ~a" i
;                 (mem-aref rat-reg-0 i)
;                 (mem-aref rat-reg-2 i))))

;   )))