(defun call-cats (cat-count wall-count wall-blck wall-step wall-start)
    (with-position-filters cat-count
      ((filter-by-walls wall-count wall-blck wall-step wall-start))
      (filter-by-walls 0.03 1.0)))
