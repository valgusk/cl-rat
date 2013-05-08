(in-package :cl-user)
(defpackage genetics
  (:use :cl
        :cl-cuda)
  (:export :main))
(in-package :genetics)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;    neural network definition   ;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;structure preparation macro helpers
(defun mem-p (layer-name) (equal (subseq (format nil "~3a" layer-name) 0 3) "MEM"))

(defun clean-gensym (a) (read-from-string (symbol-name (gensym a))))

(defun give-name (net-name layer-name suffix &optional (fun #'clean-gensym))
  (funcall fun (format nil "~a-~a-~a" net-name layer-name suffix)))

(defun assign-names (layer-name net-name)
  (flet ((gen-name (suffix) (give-name net-name layer-name suffix #'intern)))
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
    `((,inp 'float ,(* net-count inputs))
      (,out 'float ,(* net-count outputs))
      (,off 'float ,(* net-count outputs))
      (,wei 'float ,(* net-count inputs outputs)))))

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
             ((> ,i-i ,i-z))
             (set sum (+ sum (* (aref ,inp ,i-i) (aref ,wei wei-i))))
             (set wei-i (+ wei-i 1)))))))

(defun create-neuron-kernel (kernel-name inputs outputs layer input-vars all-layers is-mem)
  (destructuring-bind (inp out wei off mem) (names layer 'inp 'out 'wei 'off 'mem)
    (let ((float-vars (append (list inp out wei off)
                              (when is-mem (list mem))
                              input-vars))
          (in-count (count-inputs inputs)))
      (flet ((to-form (name) `(,name float*))
             (do-inputs (in) (process-input-var in wei all-layers inp in-count)))
        `(defkernel ,kernel-name (void ,(mapcar #'to-form float-vars))
           (let ((i (+ (* block-dim-x block-idx-x) thread-idx-x)) ;off out
                 (wei-start (* ,in-count ,(* (if is-mem 4 1) outputs) block-idx-x)) ;wei
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
               (input (aref ,mem (+ i 1)))
               (store (aref ,mem (+ i 2)))
               (give (aref ,mem (+ i 3)))
               (keep (aref ,mem (+ i 4)))
               (prev (aref ,dat o))
               (kept (* prev keep))
               (resulting (+ (* input store) kept)))
            (set (aref ,dat o) resulting)
            (set (aref ,out o) (tanh (* give resulting))))))))

;; layer function definition macro helpers
(defun get-inputs (inputs layers)
  (flet ((needed (layer) (member (first layer) (mapcar #'first inputs))))
    (mapcar #'layer-out (remove-if-not #'needed layers))))

(defun create-action (inputs outputs net-count layer all-layers is-mem)
  (destructuring-bind (inp out off wei mem dat ker ker-2 act)
                      (names layer 'inp 'out 'off 'wei 'mem 'dat 'ker 'ker-2 'act)
    (let ((input-vars (get-inputs inputs all-layers)))
      `((progn ,(create-neuron-kernel ker inputs outputs layer input-vars all-layers is-mem)
               ,@(when is-mem (create-storage-kernel ker-2 layer)))
        (,act ()
           (,ker ,inp ,out ,off ,wei
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


;execution validation
(defun calculate-neuron (mapping inp out off wei &rest additionals)
  nil)

(defun calculate-memory (mapping inp out off wei mem dat &rest additionals)
  nil)

(defun cpu-layer-action (all-layers)
  #(lambda (layer)
     (destructuring-bind (name inputs outputs) (subseq layer 0 3)
       (let* ((input-vars (get-inputs inputs all-layers))
              (mapping (mapcar #'null (mapcar #'first inputs))))
          (mapcar #(lambda (input-list)
                      (if (mem-p name)
                        `(calculate-memory ,@(names layer 'inp 'out 'off 'wei 'mem 'dat) ,@input-list)
                        `(calculate-neuron ,@(names layer 'inp 'out 'off 'wei ,@input-list))))
                  inputs)))))

(defun create-validator (layers)
  (mapcar (cpu-layer-action layers) layers))

;;neural network allocation and definition
(defmacro with-neural-networks (name count layers &body body)
  (let* ((layers (add-var-names layers name))
         (allocation-list (allocate-net-memory count layers))
         (kernels-actions (create-net-actions count layers)))
    (format t "~%The ~a will require ~a MB on host and device~%"
              name
              (/ (* 4.0 (apply #'+ (mapcar #'third allocation-list))) 1024 1024))
    (progn `(with-memory-blocks ,allocation-list
              ,@(mapcar #'first kernels-actions)
              (labels (,@(mapcar #'second kernels-actions)
                       (,(read-from-string (format nil "run-~a" name)) ()
                        ,@(mapcar #'(lambda (layer) (names layer 'act)) layers)))
                ,@body)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;     neural netwwork data manipulation      ;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;     main application code      ;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun main ()
  (with-cuda-context (0)
    (with-neural-networks rat
                          4096
                          ;name   inputs                outputs
                          ((A     ((nil 0 64) (F 64 96))    96)
                           (B     ((A 0 96))                96)
                           (C     ((B 0 96))                96)
                           (MEM-D ((C 0 96))                32)
                           (E     ((C 0 96) (MEM-D 0 32))   96)
                           (F     ((E 0 96))                96)
                           (G     ((F 0 96))                96))
      (dotimes (i 10 T) (run-rat)))))