;; Caution: avoid using dynamic-arity atom as list, as it will be expensive. Used Cons cell instead


(: length-atom (-> Expression Number))
(= (length-atom $expr) 
    (if (noreduce-eq $expr ())
        0
        (let $rest (cdr-atom $expr)
        (+ 1 (length-atom $rest))
        )
    )
)

(: index-atom (-> Expression Number Atom))
(= (index-atom $expr $idx)
    (if (or (< $idx 0) 
            (and (noreduce-eq $expr ()) (>= $idx 0)))
        (Error $expr "Index out of range")
        (let 
            ($head $rest) (decons-atom $expr)
            (if (== $idx 0)
                $head
                (index-atom $rest (- $idx 1))
            )
        )    
    )
)

;; take O(n), caution when use
(: append-atom (-> Expression Atom Expression))
(= (append-atom $expr $a) 
    (if (noreduce-eq $expr ())
        ($a)
        (let* (
            ($head (car-atom $expr))
            ($tail (cdr-atom $expr))
            ($rest (append-atom $tail $a))
        ) (cons-atom $head $rest))
    ) 
)

(: slice-atom (-> Expression Number (Expression Expression)))
(= (slice-atom $expr $n)
    (if (or (< $n 0) 
            (and (noreduce-eq $expr ()) (> $n 0)))
        (Error $expr "Index out of range")
        (if (== $n 0) 
            (() $expr)
            (let* (
                (($head $tail) (decons-atom $expr))
                (($rhead $rtail) (slice-atom $tail (- $n 1))) 
            ) ((cons-atom $head $rhead) $rtail))
        )
    )
)

(: reverse-atom (-> Expression Expression))
(= (reverse-atom $expr)
    (reverse-atom-helper $expr ())
)

(: reverse-atom-helper (-> Expression Expression Expression))
(= (reverse-atom-helper $expr $acc)
    (if (noreduce-eq $expr ())
        $acc
        (let* (
            (($head $rest) (decons-atom $expr))
            ($updated-acc (cons-atom $head $acc))  
        ) (reverse-atom-helper $rest $updated-acc))
    )
)

(: extend-atom (-> Expression Expression Expression))
(= (extend-atom $a $b)
    (let $ar (reverse-atom $a)
        (foldl-atom $ar $b $acc $head (cons-atom $head $acc))
    )
)

;; FIXME: scopeless implementation of lambda
(: lambda (-> Atom $t (-> $a $t)))
(= ((lambda $var $body) $arg)
   (let $var $arg $body))
; !((lambda $x $x) (lambda $x $x))
; !(((lambda $x (lambda $x $x)) 1) 2)
; !(let $x 0
;     (let* (
;         (() (println! $x))
;         ($x 2)
;         ($y 3)
;     ) (ret $x $y))
; )


(: print-lines! (-> Expression (->)))
(= (print-lines! $ls) 
   (if (== (quote $ls) (quote ()))
    ()
    (let* (
        ($head (car-atom $ls))
        (() (println! $head))
        ($rest (cdr-atom $ls))
    ) (print-lines! $rest))
   )
)

;; TODO: strongly-typed list of numbers
;; left-inclusive, right-exclusive (start, end]
(: arange (-> Number Number Number Expression))
(= (arange $start $end $step) 
    (arange-helper $start $end $step ())
)
(: arange-helper (-> Number Number Number Expression Expression))
(= (arange-helper $start $end $step $acc)
    (if (< $start $end)
        (let* (
          ($next (+ $start $step))
          ($acc-new (cons-atom $start $acc)) )     
          (arange-helper $next $end $step $acc-new)
        )
        (reverse-atom $acc)
    )
)

(: arange (-> Number Number Expression))
(= (arange $start $end)
    (arange $start $end 1)
)

(: string-concat (-> String String String))
(= (string-concat $a $b) 
    (let* (
        ($chars-a (stringToChars $a))
        ($chars-b (stringToChars $b))
        ($chars-ab (extend-atom $chars-a $chars-b))
    ) (charsToString $chars-ab))
)

;; make a variable from number
(: make-var (-> Number String))
(= (make-var $id)
    (parse (string-concat "$" (repr $id)))
)

;; make a function with $arity parameters
(: make-func (-> Symbol Number Expression))
(= (make-func $functor $arity)
    (make-func-helper $functor $arity ())
)

(: make-func-helper (-> Symbol Number Expression Expression))
(= (make-func-helper $functor $arity-left $accum)
    (if (== $arity-left 0)
        (cons-atom $functor $accum)
        (let* (
            ($var (make-var $arity-left))
            ($accum-new (cons-atom $var $accum))
        ) (make-func-helper $functor (- $arity-left 1) $accum-new)) 
    )
)


;; TODO: needs more testing, is this approach sound?
;; ((f $1 $2) $3) --> (f $1 $2 $3), Eat up arguments up to a certain amount
;; variable names range from $1 --> $<max-arity>
;; this modifies space
(: make-uncurry! (-> Grounded Symbol Number Number (->)))
(= (make-uncurry! $space $functor $min-arity $max-arity)
    (if (== $min-arity $max-arity)
        (empty)
        (let* (
            ($this-func (make-func $functor $min-arity))
            ($next-arity (+ 1 $min-arity))
            ($next-func (make-func $functor $next-arity))
            ($var-to-eat (make-var $next-arity))
            ($eqn (= ($this-func $var-to-eat) $next-func))
            (() (add-atom $space $eqn))
        ) (make-uncurry! $space $functor $next-arity $max-arity))
    )
)


;; for mettalog
!(if (== (collapse (match &self (= (noreduce-eq $x $y) $t) True)) ())
    (trace! ("added noreduce-eq") (add-atom &self 
        (= (noreduce-eq $x $y) (== (quote $x) (quote $y)))
    ))
    ()
)

!(if (== (>= 2 1) True)
    ()
    (trace! ("add greater or equal sign")
        (add-atom &self 
            (= (>= $x $y) (not (< $x $y)))
        )
    )
)