;; 

;; generate all possible programs with a given set of combinators

!(import! &self lib)

;; (program $depth $prog)
;; non-deterministic semantics make the logic very elegant

;; add & return progn if not exists
(= (add-progn-if-nexists! $depth $progn)
    (if (== (collapse (match &self (program $depth $progn) exists)) ())    
        (let () (add-atom &self (program $depth $progn))
            $progn
        )
        (empty)
    )   
)

(= (generate! $d $combinators)
    (if (> $d 1)
        (let* (
            ($last-progn (generate! (- $d 1) $combinators))
            ($cur (superpose $combinators))
            ($nxt-prog-1 ($cur $last-progn))
            ($nxt-prog-2 ($last-progn $cur))
            ; (() (println! (at $d $nxt-prog-1 $nxt-prog-2)))
            
        ) (superpose (
            (add-progn-if-nexists! $d $nxt-prog-1)
            (add-progn-if-nexists! $d $nxt-prog-2)
        )))
        (superpose $combinators)
    )
)
!(generate! 3 (S K))
; !(add-progn-if-nexists! 3 (S K))
!(match &self (program 3 $x) $x)