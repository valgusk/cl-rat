;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;    basement generation code    ;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defstruct basement rats plants cats walls)

(load "basement-utils.lisp")
(load "walls.lisp")
(load "plants.lisp")
(load "cats.lisp")


(defun make-basements (count &key (wall-count 10) (wall-len 50))
  (loop for i below count
        for bound-walls = `((0 0 ,0 100)
                            (0 0 ,(* PI 1.5) 100)
                            (99 0 ,(* PI 1.5) 100)
                            (0 99 0 100))
        for walls = (apply #'append (make-walls wall-count wall-len bound-walls))
        collect (make-basement :walls walls)))


; (defun call-cats (basement)
;   (loop repeat 4 do
;     (loop do
;       (let* ((x (+ 20 (random 80)))
;              (y (+ 20 (random 80)))
;              (cat `(x ,x y ,y type #\C health 1.0)))
;         (when (notany (neighbour-tester cat 4 T)
;                       (append (basement-walls basement)
;                               (basement-cats basement)))
;           (push cat (basement-cats basement))
;           (return))))))


; (defun grow-plants (basement)
;   (loop repeat 200 do
;     (loop do
;       (let* ((x (random 100))
;              (y (random 100))
;              (plant `(x ,x y ,y type #\@ health 1.0)))
;         (when (or (some (neighbour-tester plant)
;                         (append (basement-walls basement)
;                                 (basement-plants basement)))
;                   (< (random 1.0) 0.01))
;           (push plant (basement-plants basement))
;           (return))))))


; (defun create-random-basement ()
;   (let ((basement (make-basement)))
;     (setf (basement-walls basement) (build-walls))
;     (grow-plants basement)
;     (call-cats basement)
;     basement))

; (defun show-objects (basement)
;   (let ((rows (loop for i from 0 to 100 collect (make-string 100 :initial-element #\Space))))
;     (loop for w in (append (basement-plants basement)
;                            (basement-walls basement)
;                            (basement-cats basement)) do
;       (setf (elt (nth (getf w 'x) rows) (getf w 'y)) (getf w 'type)))
;   (format t "~{~a~^~%~}" rows)))

; ; (show-objects (create-random-basement))

; (defun find-available-basement (basements)
;   (let ((ok-basements (loop for b in basements
;                          if (< (length (basement-rats b)) 40)
;                          collect b))
;         (available-basements (loop for b in basements
;                                if (< (length (basement-rats b)) 50)
;                                collect b)))
;     (if (< (length ok-basements) 3)
;         (nth (random (length available-basements)) available-basements)
;         (nth (random (length ok-basements)) ok-basements))))