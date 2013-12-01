; create walls, manually defined walls go to existing and
; should have format: (x y angle length)
(defun make-walls (&optional (count 100) (len 50) (existing nil))
  (labels
    ((wall-def nil (mapcar #'random `(100 100 ,(* PI 2) ,len)))
     (stick-p (a b) (> 2 (apply #'max (mapcar #'abs (mapcar '- a b)))))
     (by-x (angle) (member (floor (/ angle PI 1/4)) '(0 3 4 7)))
     (sticks (a) (lambda (bs) (find a bs :test #'stick-p)))
     (make-grps (crosses grps &optional (i 0) (grp (nth i grps)))
       (cond ((null crosses) grps)
             ((null (car crosses)) (make-grps (cdr crosses) grps (1+ i)))
             ((zerop grp) nil)
             (t (make-grps (cdr crosses) (substitute 0 grp grps) (1+ i)))))
     (unblock-wall (fixed w walls &optional grps ok-bricks)
       (symbol-macrolet
         ((new-grps (make-grps (mapcar (sticks (car w)) walls) grps))
          (bad-p (when walls (not new-grps)))
          (ok-grps (if (or fixed ok-bricks) (mapcar #'1+ (cons 0 grps)) grps))
          (ok-walls (if ok-bricks (cons ok-bricks walls) walls)))
         (cond (fixed (values (cons w walls) ok-grps))
               ((null w) (values ok-walls ok-grps))
               (bad-p (unblock-wall nil (cdr w) ok-walls ok-grps))
               (t (unblock-wall nil (cdr w) walls new-grps (cons (car w) ok-bricks))))))
     (make-bricks (x y angle size)
       (loop for i below size
             for x1 = x then (if (by-x angle) (1+ x1) (+ x (* i (cos angle))))
             for y1 = y then (if (by-x angle) (+ y (* i (sin angle))) (1+ y1))
             when (>= 99.4 (max x1 y1) (min x1 y1) -0.5)
             collect (list (round x1) (round y1))))
     (make-all-bricks (existing left)
       (multiple-value-call #'unblock-wall (nth left existing)
         (apply #'make-bricks (or (nth left existing) (wall-def)))
         (and (plusp left) (make-all-bricks existing (1- left))))))
    (make-all-bricks existing (+ count (list-length existing)))))


;; move basement data to gfx card
(defun basements-walls-to-device (basements wall-blck wall-step)
  (loop for (x y) in (apply #'append (mapcar #'basement-walls basements))
        for i = 0 then (+ wall-step i) do
          (setf (mem-aref wall-blck i) (coerce x 'float)
                (mem-aref wall-blck (1+ i)) (coerce y 'float)
                (mem-aref wall-blck (+ 2 i)) 1.0) ;health
        finally (return (memcpy-host-to-device wall-blck))))
