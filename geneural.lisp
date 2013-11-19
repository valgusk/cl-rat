
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;       some dirty hacks         ;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(setf (getf cl-cuda::+built-in-functions+ 'tanh) '(((float) float nil "tanh")))
(setf (getf cl-cuda::+built-in-functions+ 'nearbyintf) '(((float) float nil "nearbyintf")))
(setf (getf cl-cuda::+built-in-functions+ 'fminf) '(((float float) float nil "fminf")))
(setf (getf cl-cuda::+built-in-functions+ 'fmaxf) '(((float float) float nil "fmaxf")))
(setf (getf cl-cuda::+built-in-functions+ 'fmodf) '(((float float) float nil "fmodf")))
(setf (getf cl-cuda::+built-in-functions+ 'fabsf) '(((float) float nil "fabsf")))
(setf (getf cl-cuda::+built-in-functions+ 'sinf) '(((float) float nil "sinf")))
(setf (getf cl-cuda::+built-in-functions+ 'cosf) '(((float) float nil "cosf")))
(setf (getf cl-cuda::+built-in-functions+ 'atan2f) '(((float float) float nil "atan2f")))
(setf (getf cl-cuda::+built-in-functions+ 'to-int) '(((float) int nil "__float2int_rd")))
(setf (getf cl-cuda::+built-in-functions+ 'to-float) '(((int) float nil "__int2float_rd")))
(setf (getf cl-cuda::+built-in-functions+ 'copysignf) '(((float float) float nil "copysignf")))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;    neural network definition   ;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;structure preparation macro helpers
(defun mem-p (layer-name) (equal (subseq (format nil "~3a" layer-name) 0 3) "MEM"))

(defun clean-gensym (a) (read-from-string (symbol-name (gensym a))))

(defun give-name (params &optional (fun #'read-from-string))
  (funcall fun (format nil "~{~a~^-~}" params)))

(defun assign-names (layer-name net-name)
  (flet ((gen-name (suffix) (give-name (list net-name layer-name suffix))))
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
    (loop for n from 0 below count collect
      `(,(give-name (list name 'reg n))
        'float
        ,(reduce #'+ (mapcar #'third (allocate-net-memory 1 layers))))))

;; layer kernel definition macro helpers
(defun process-input-var (input-list wei all-layers def-name def-width)
  (destructuring-bind (name start end) input-list
    (let* ((i-a (clean-gensym "i-a"))
           (i-z (clean-gensym "i-z"))
           (i-i (clean-gensym "i-i"))
           (i-layer (find name all-layers :key #'first))
           (i-width (or (third i-layer) def-width))
           (inp (or (layer-out i-layer) def-name)))
      `(let* ((,i-a (+ ,start (* ,i-width block-idx-x)))
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
           (let* ((i (+ (* block-dim-x block-idx-x) thread-idx-x)) ;off out
                  (wei-start (* ,in-count (+ (* block-dim-x block-idx-x) thread-idx-x))) ;wei
                  (wei-i wei-start)
                  (sum 0.0))
             ,@(mapcar #'do-inputs inputs)
             (set sum (+ sum (aref ,off i)))
             (set (aref ,(if is-mem mem out) i) (tanh sum))))))))

(defun create-storage-kernel (kernel-name layer)
  (destructuring-bind (mem out dat) (names layer 'mem 'out 'dat)
    `((defkernel ,kernel-name (void ((,mem float*) (,out float*) (,dat float*)))
         (let* ((i (+ (* block-dim-x block-idx-x 4) (* 4 thread-idx-x)))
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
(load "validation.lisp")
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
             (init-fill ,off)
             ,@(when (mem-p (first layer))
                 `((init-fill ,mem #'(lambda () 0.0))
                   (init-fill ,dat #'(lambda () 0.0)))))))
     layers))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;        neural network transport actions       ;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defun copy-code (param size d-to-h)
  (let* ((start (give-name (list "start" param) #'clean-gensym))
         (device `(aref ,param (+ ,start i)))
         (host `(aref chromosome (+ cur i))))
    `(let ((,start (* id ,size)))
        (if (< i ,size)
          (set ,@(if d-to-h `(,host ,device) `(,device ,host))))
        (set cur (+ cur ,size)))))

(defun add-chromosome-transport (name layers d-to-h)
  (flet ((params (l) (allocate-layer-memory l 1))
         (copy-code (param size) (copy-code param size d-to-h)))
    (let* ((kernel-name (give-name (list name (if d-to-h 'dissect 'stitch) 'kernel)))
           (layer-data (mapcan #'params layers))
           (ps (loop for l in layer-data collect `(,(first l) float*)))
           (pnames (loop for l in layer-data collect (first l)))
           (psizes (loop for l in layer-data collect (third l))))
      `((defkernel ,kernel-name (void (,@ps (chromosome float*) (id int)))
           (let ((cur 0)
                 (i (+ thread-idx-x (* block-idx-x block-dim-x))))
             ,@(mapcar #'copy-code pnames psizes)))
         (,(give-name (list name (if d-to-h 'dissect 'stitch))) (id blk)
            (,kernel-name ,@pnames blk id
              :grid-dim (list ,(ceiling (/ (apply #'max psizes) 512)) 1 1)
              :block-dim (list 512 1 1)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;      neural network genetic modifications       ;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun extend-options (options off)
  (when options
    (let ((end (+ off (third (first options)))))
      (cons (append (first options) `(,off ,end))
            (extend-options (rest options) end)))))

(defun each-layer (fun layers &optional (off 0))
  (let* ((l (first layers))
         (options (extend-options (allocate-layer-memory l 1) off))
         (offs (find (car (names l 'off)) options :key #'first))
         (weis (find (car (names l 'wei)) options :key #'first))
         (end (+ off (fifth (first (last options))))))
    `(,@(funcall fun weis offs (set-difference options (list weis offs)))
      ,@(when (rest layers) (each-layer fun (rest layers) end)))))

(defun add-chromosome-action (name layers type)
  (labels
    ((iterate-genetic (wei off rest)
       `((do ((wei ,(fourth wei) (+ wei ,(/ (third wei) (third off))))
              (off ,(fourth off) (1+ off)))
             ((= off ,(fifth off)) nil)
           (act wei off (+ wei ,(/ (third wei) (third off)))))
         ,@(nullify-rest (merge-rest rest))))
     (merge-rest (rest)
        (when rest (if (equal (fourth (car rest)) (fifth (cadr rest)))
                       (merge-rest
                         (cons (append (butlast (cadr rest)) (last (car rest)))
                               (cddr rest)))
                       (cons (car rest) (merge-rest (cdr rest))))))
     (nullify-rest (rest)
        (loop for r in rest collect
          `(loop for i from ,(fourth r) below ,(fifth r) do
              (setf (mem-aref result i) 0.0)))))
    (let ((crossover
           `(let ((par (if (< (random 1.0) 0.5) parent-b parent-a)))
              (setf (mem-aref result off) (mem-aref par off))
              (loop for i from wei below next-wei do
                (setf (mem-aref result i) (mem-aref par i)))))
          (mutation
           `(let ((should-rand (< (random 1.0) 0.2)))
              (setf (mem-aref result off)
                    (+ (mem-aref parent-a off) (if should-rand (1- (random 2.0)) 0)))
              (loop for i from wei below next-wei do
                (setf (mem-aref result i)
                    (+ (mem-aref parent-a i) (if should-rand (1- (random 2.0)) 0)))))))
      (declare (special crossover mutation))
      `(,(give-name (list name type)) (result parent-a &optional parent-b)
          (declare (ignorable parent-b))
          (flet ((act (wei off next-wei) ,(symbol-value type)))
          ,@(each-layer #'iterate-genetic layers))))))

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
    (progn `(with-memory-blocks (,@allocation-list ,@register-list)
              ,@(mapcar #'first kernels-actions)
              ,@(add-initialization layers)
              ,(first dissect-kernel-action)
              ,(first stitch-kernel-action)
              (labels ((,(give-name (list name 'count)) nil ,count)
                       ,@(mapcar #'second kernels-actions)
                       (,(read-from-string (format nil "run-~a" name)) ()
                       ,@(mapcar #'(lambda (layer) (names layer 'act)) layers))
                       ,(second dissect-kernel-action)
                       ,(second stitch-kernel-action)
                       ,(add-chromosome-action name layers 'crossover)
                       ,(add-chromosome-action name layers 'mutation)
                       (validate ()
                         ,@(create-validator layers count)))
                ,@body)))))
