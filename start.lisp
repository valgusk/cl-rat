(in-package :cl-user)
(defpackage genetics
  (:use :cl
        :cl-cuda)
  (:export :main))


(in-package :genetics)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;       some dirty hacks         ;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(setf (getf cl-cuda::+built-in-functions+ 'tanh) '(((float) float nil "tanh")))
(setf (getf cl-cuda::+built-in-functions+ 'nearbyintf) '(((float) float nil "nearbyintf")))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;    neural network definition   ;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;structure preparation macro helpers
(defun mem-p (layer-name) (equal (subseq (format nil "~3a" layer-name) 0 3) "MEM"))

(defun clean-gensym (a) (read-from-string (symbol-name (gensym a))))

(defun give-name (params &optional (fun #'clean-gensym))
  (funcall fun (format nil "~{~a~^-~}" params)))

(defun assign-names (layer-name net-name)
  (flet ((gen-name (suffix) (give-name (list net-name layer-name suffix) #'read-from-string)))
    (mapcar #'gen-name '(inp out wei off mem dat ker ker-2 act))))

(defun add-var-names (layers net-name)
  (flet ((add-vars (layer) (append layer `(,(assign-names (first layer) net-name)))))
    (mapcar #'add-vars layers)))

(defun names (layer &rest needed)
  (mapcar #'(lambda (need)
              (nth (position need '(inp out wei off mem dat ker ker-2 act))
                   (fourth layer)))
          needed))

(defun count-inputs (inputs)
  (apply #'+ (mapcar #'(lambda (i) (- (third i) (second i))) inputs)))

(defun layer-out (layer) (car (names layer 'out)))

;; memory allocation macro helpers
(defun allocate-neuron (inputs outputs net-count layer)
  (destructuring-bind (inp out off wei) (names layer 'inp 'out 'off 'wei)
    (let ((outputs-2 (if (mem-p (car layer)) (* 4 outputs) outputs)))
      `((,inp 'float ,(* net-count inputs))
        (,out 'float ,(* net-count outputs))
        (,off 'float ,(* outputs-2 net-count))
        (,wei 'float ,(* net-count inputs outputs-2))))))

(defun allocate-storage (inputs outputs net-count layer)
  (let ((gate-layer (allocate-neuron inputs outputs net-count layer)))
    (destructuring-bind (mem dat) (names layer 'mem 'dat)
      (append `((,mem 'float ,(* net-count 4 outputs))
                (,dat 'float ,(* net-count outputs))) gate-layer))))

(defun allocate-layer-memory (layer net-count)
  (destructuring-bind (layer-name inputs outputs) (subseq layer 0 3)
      (apply (if (mem-p layer-name) #'allocate-storage #'allocate-neuron)
             (list (count-inputs inputs) outputs net-count layer))))

(defun layer-maker (net-count)
  #'(lambda (layer) (allocate-layer-memory layer net-count)))

(defun allocate-net-memory (count layers)
  (mapcan (layer-maker count) layers))

(defun allocate-register-memory (name count layers)
  (let ((size (loop for l in layers sum
                (* (third l)(+ (count-inputs (second l)) 1)))))
    (loop for n from 0 below count collect
      `(,(give-name (list name 'reg n) #'read-from-string) 'float ,size))))

;; layer kernel definition macro helpers
(defun process-input-var (input-list wei all-layers def-name def-width)
  (destructuring-bind (name start end) input-list
    (let* ((i-a (clean-gensym "i-a"))
           (i-z (clean-gensym "i-z"))
           (i-i (clean-gensym "i-i"))
           (i-layer (find name all-layers :key #'first))
           (i-width (or (third i-layer) def-width))
           (inp (or (layer-out i-layer) def-name)))
      `(let ((,i-a (+ ,start (* ,i-width block-idx-x)))
             (,i-z (+ ,i-a ,(- end start))))
         (do ((,i-i ,i-a (+ ,i-i 1)))
             ((>= ,i-i ,i-z))
             (set sum (+ sum (* (aref ,inp ,i-i) (aref ,wei wei-i))))
             (set wei-i (+ wei-i 1)))))))

(defun create-neuron-kernel (kernel-name inputs layer input-vars all-layers is-mem)
  (destructuring-bind (inp out wei off mem) (names layer 'inp 'out 'wei 'off 'mem)
    (let ((float-vars (append (list inp out wei off)
                              (when is-mem (list mem))
                              input-vars))
          (in-count (count-inputs inputs)))
      (flet ((to-form (name) `(,name float*))
             (do-inputs (in) (process-input-var in wei all-layers inp in-count)))
        `(defkernel ,kernel-name (void ,(mapcar #'to-form float-vars))
           (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)) ;off out
                 (wei-start (* ,in-count (+ (* block-dim-x block-idx-x) thread-idx-x))) ;wei
                 (wei-i wei-start)
                 (sum 0.0))
             ,@(mapcar #'do-inputs inputs)
             (set sum (+ sum (aref ,off i)))
             (set (aref ,(if is-mem mem out) i) (tanh sum))))))))

(defun create-storage-kernel (kernel-name layer)
  (destructuring-bind (mem out dat) (names layer 'mem 'out 'dat)
    `((defkernel ,kernel-name (void ((,mem float*) (,out float*) (,dat float*)))
         (let ((i (+ (* block-dim-x block-idx-x) (* 4 thread-idx-x)))
               (o (+ (* block-dim-x block-idx-x) thread-idx-x))
               (input (aref ,mem i))
               (store (nearbyintf (/ (+ 1.0 (aref ,mem (+ i 1))) 2.0)))
               (give (nearbyintf (/ (+ 1.0 (aref ,mem (+ i 2))) 2.0)))
               (keep (nearbyintf (/ (+ 1.0 (aref ,mem (+ i 3))) 2.0)))
               (prev (aref ,dat o))
               (kept (* prev keep))
               (resulting (+ (* input store) kept)))
            (set (aref ,dat o) resulting)
            (set (aref ,out o)  (tanh (* give resulting))))))))

;; layer function definition macro helpers
(defun get-inputs (inputs layers)
  (flet ((needed (layer) (member (first layer) (mapcar #'first inputs))))
    (mapcar #'layer-out (remove-if-not #'needed layers))))

(defun create-action (inputs outputs net-count layer all-layers is-mem)
  (destructuring-bind (inp out off wei mem dat ker ker-2 act)
                      (names layer 'inp 'out 'off 'wei 'mem 'dat 'ker 'ker-2 'act)
    (let ((input-vars (get-inputs inputs all-layers)))
      `((progn ,(create-neuron-kernel ker inputs layer input-vars all-layers is-mem)
               ,@(when is-mem (create-storage-kernel ker-2 layer)))
        (,act ()
           (,ker ,inp ,out ,wei ,off
                 ,@(when is-mem (list mem))
                 ,@input-vars
                 :grid-dim (list ,net-count 1 1)
                 :block-dim (list ,(* (if is-mem 4 1) outputs) 1 1))
           ,@(when is-mem
               `((,ker-2 ,mem ,out ,dat
                         :grid-dim (list ,net-count 1 1)
                         :block-dim (list ,outputs 1 1)))))))))

(defun create-layer-action (layer net-count all-layers)
  (destructuring-bind (layer-name inputs outputs) (subseq layer 0 3)
    (create-action inputs outputs net-count layer all-layers (mem-p layer-name))))

(defun action-maker (net-count all-layers)
  #'(lambda (layer) (create-layer-action layer net-count all-layers)))

(defun create-net-actions (count layers)
  (mapcar (action-maker count layers) layers))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;    neural network execution validation   ;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
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
      (test out cpu-results)))

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
    (print (test mem gate-results))
    (print 'memory-outputs)
    (print (test out cpu-results))))


(defun cpu-layer-action (all-layers count)
  #'(lambda (layer)
     (destructuring-bind (name inputs outputs) (subseq layer 0 3)
       (let* ((input-vars (get-inputs inputs all-layers))
              (mapping (mapcar #'(lambda(i)
                                   (cons (third (find (or (first i) name) all-layers :key #'first)) i))
                               inputs)))
          (if (mem-p name)
            `(validate-memory ,(min 1 count) ,outputs ',mapping #',@(names layer 'act)
                ,@(names layer 'inp 'out 'off 'wei 'mem 'dat) ,@input-vars)
            `(print (validate-neuron ,(min 1 count) ,outputs ',mapping #',@(names layer 'act)
                ,@(names layer 'inp 'out 'off 'wei) ,@input-vars)))))))

(defun create-validator (layers count)
  (mapcar (cpu-layer-action layers count) layers))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;      neural network data initialization    ;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun init-fill (memory-block &optional (fun #'(lambda () (1- (random 2.0)))))
  (loop for n from 0 below (cl-cuda::memory-block-cuda-length memory-block) do
    (setf (mem-aref memory-block n) (funcall fun)))
  (memcpy-host-to-device memory-block))

(defun add-initialization (layers)
  (mapcan
     #'(lambda (layer)
         (destructuring-bind (inp out wei off dat mem) (names layer 'inp 'out 'wei 'off 'dat 'mem)
           `((init-fill ,inp)
             (init-fill ,out #'(lambda () 0.0))
             (init-fill ,wei)
             (init-fill ,off #'(lambda () 1.0))
             ,@(when (mem-p (first layer))
                 `((init-fill ,mem #'(lambda () 0.0))
                   (init-fill ,dat #'(lambda () 0.0)))))))
     layers))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;      neural network runtime modifications       ;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun crossover (net-name layers)
  (let ((cross (read-from-string (format nil "cross-~a" name))))
    `(,cross (id-a id-b id-result)
          nil)))

(defun mutation (net-name layers)
  (let ((mutate (read-from-string (format nil "mutate-~a" name))))
    `(,mutate (id id-result)
          nil)))

(defun add-chromosome-transport (name layers d-to-h)
  (flet ((params (l)
            (let ((wei-n (* (third l) (count-inputs (second l))))
                  (off-n (third l)))
              (destructuring-bind (wei off) (names l 'wei 'off)
                `(((,wei float*) (,off float*))
                  (,wei ,off)
                  (,wei-n ,off-n)))))
         (copy-code (param size)
            (let* ((start (give-name (list "start" param)))
                   (device `(aref ,param (+ ,start i)))
                   (host `(aref chromosome (+ cur i))))
              `(let ((,start (* id ,size)))
                  (if (< i ,size)
                    (set ,@(if d-to-h `(,device ,host) `(,host ,device))))
                  (set cur (+ cur ,size))))))
    (let* ((kernel-name (give-name (list name (if d-to-h 'dissect 'stitch) 'kernel)
                                   #'read-from-string))
           (layer-data (mapcar #'params layers))
           (ps (mapcan #'first layer-data))
           (pnames (mapcan #'second layer-data))
           (psizes (mapcan #'third layer-data)))
      `((defkernel ,kernel-name (void (,@ps (chromosome float*) (id int)))
           (let ((cur 0)
                 (i (+ thread-idx-x (* block-idx-x block-dim-x))))
             ,@(mapcar #'copy-code pnames psizes)))
         (,(give-name (list name (if d-to-h 'dissect 'stitch)) #'read-from-string) (id blk)
            (,kernel-name ,@pnames blk id))))))


(defun add-chromosome-actions (layers)
  nil)



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;      neural network definition macro       ;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;neural network allocation and definition
(defmacro with-neural-networks (name count layers &body body)
  (let* ((layers (add-var-names layers name))
         (allocation-list (allocate-net-memory count layers))
         (kernels-actions (create-net-actions count layers))
         (dissect-kernel-action (add-chromosome-transport name layers T))
         (stitch-kernel-action (add-chromosome-transport name layers nil))
         (register-list (allocate-register-memory name 3 layers)))
    (format t "~%The ~a will require ~a MB on host and device~%"
              name
              (/ (* 4.0 (apply #'+ (mapcar #'third allocation-list))) 1024 1024))
    (print `(with-memory-blocks (,@allocation-list ,@register-list)
              ,@(mapcar #'first kernels-actions)
              ,@(add-initialization layers)
              ,(first dissect-kernel-action)
              ,(first stitch-kernel-action)
              (labels (,@(mapcar #'second kernels-actions)
                       (,(read-from-string (format nil "run-~a" name)) ()
                       ,@(mapcar #'(lambda (layer) (names layer 'act)) layers))
                       ,(second dissect-kernel-action)
                       ,(second stitch-kernel-action)
                       (validate ()
                         ,@(create-validator layers count)))
                ,@body)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;     main application code      ;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun main ()
  (with-cuda-context (0)
    (with-neural-networks rat
                          2
                          ;name   inputs                outputs
                          ((A     ((nil 0 64) (G 64 96))    96)
                           (B     ((A 0 96))                96)
                           (C     ((B 0 96))                96)
                           (MEM-D ((C 0 96))                32)
                           (E     ((C 0 96) (MEM-D 0 32))   96)
                           (F     ((E 0 96))                96)
                           (G     ((F 0 96))                96))
      (validate))))

(main)