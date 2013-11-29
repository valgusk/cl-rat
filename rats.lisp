(defun throw-rats (rat-count wall-count wall-blck wall-step wall-start
                             cat-count cat-blck cat-step cat-start
                             rat-count rat-blck rat-step rat-start)
    (with-position-filters rat-count
      ((filter-by-walls wall-count wall-blck wall-step wall-start)
       (filter-by-cats cat-count cat-blck cat-step cat-start)
       (filter-by-rats rat-count rat-blck rat-step rat-start))
      (filter-by-walls 0.01 1.0)
      (filter-by-cats 0.2 1.0)
      (filter-by-rats 0.01 1.0)))
