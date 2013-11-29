(defun grow-plants (plant-count wall-count wall-blck wall-step wall-start)
    (with-position-filters plant-count
      ((filter-by-walls wall-count wall-blck wall-step wall-start))
      (filter-by-walls 0.01 1.0)
      (filter-by-walls 0.1 -1.0)))
