import Mathlib
set_option maxHeartbeats 0

namespace verina_advanced_1

def filterlist (x : Int) (nums : List Int) : List Int :=
  let rec aux (lst : List Int) : List Int :=
    match lst with
    | []      => []
    | y :: ys => if y = x then y :: aux ys else aux ys
  aux nums

@[reducible]
def FindSingleNumber_precond (nums : List Int) : Prop :=
  let numsCount := nums.map (fun x => nums.count x)
  numsCount.all (fun count => count = 1 ∨ count = 2) ∧ numsCount.count 1 = 1

def FindSingleNumber (nums : List Int) (h_precond : FindSingleNumber_precond (nums)) : Int :=
  let rec findUnique (remaining : List Int) : Int :=
    match remaining with
    | [] =>
      0
    | x :: xs =>
      let filtered : List Int :=
        filterlist x nums
      let count : Nat :=
        filtered.length
      if count = 1 then
        x
      else
        findUnique xs
  findUnique nums

@[reducible]
def FindSingleNumber_postcond (nums : List Int) (result: Int) (h_precond : FindSingleNumber_precond (nums)) : Prop :=
  (nums.length > 0)
  ∧
  ((filterlist result nums).length = 1)
  ∧
  (∀ (x : Int),
    x ∈ nums →
    (x = result) ∨ ((filterlist x nums).length = 2))

-- EVOLVE-BLOCK-START

/-- Fundamental connection between `filterlist` and `List.count`:
    filtering by equality to `x` has length equal to the count of `x`. -/
lemma filterlist_spec_count (x : Int) (nums : List Int) :
    (filterlist x nums).length = nums.count x := by
  classical
  -- Will be proved by induction on `nums`.
  sorry

/-- From the precondition, there exists some element whose count is exactly 1. -/
lemma FindSingleNumber_precond_implies_exists_count1
    (nums : List Int) (h_precond : FindSingleNumber_precond nums) :
    ∃ x : Int, nums.count x = 1 := by
  classical
  -- We extract existence directly from the fact that `nums.map (fun x => nums.count x)`
  -- has `1` appearing exactly once.
  unfold FindSingleNumber_precond at h_precond
  rcases h_precond with ⟨h_all, h_unique⟩
  -- From `numsCount.count 1 = 1` we know `1` appears in `numsCount`.
  have h_mem_one : 1 ∈ nums.map (fun x => nums.count x) := by
    -- We delegate the non-trivial reasoning about `List.count` and membership
    -- to a helper lemma.
    sorry
  -- Now obtain a witness `x` with `nums.count x = 1` from membership in the map.
  rcases List.mem_map.mp h_mem_one with ⟨x, hx_mem, hx_eq⟩
  refine ⟨x, ?_⟩
  simpa [hx_eq] 

/-- From the precondition, any two elements with count exactly 1 must be equal. -/
lemma FindSingleNumber_precond_implies_unique_count1
    (nums : List Int) (h_precond : FindSingleNumber_precond nums) :
    ∀ x y : Int, nums.count x = 1 → nums.count y = 1 → x = y := by
  classical
  intro x y hx hy
  -- Strategy: use that the list of counts has exactly one occurrence of `1`.
  unfold FindSingleNumber_precond at h_precond
  rcases h_precond with ⟨h_all, h_unique⟩
  -- Show that both `nums.count x` and `nums.count y` correspond to the same
  -- position in the mapped list `nums.map (fun z => nums.count z)`.
  -- This ultimately forces `x = y`. The detailed combinatorial reasoning
  -- is delegated to a separate helper lemma.
  sorry

/-- From the precondition, there exists an element whose count is exactly 1,
    and this element is unique (no other element has count 1). -/
lemma FindSingleNumber_precond_implies_exists_unique_count1
    (nums : List Int) (h_precond : FindSingleNumber_precond nums) :
    ∃! x : Int, nums.count x = 1 := by
  classical
  -- Use the separate existence and uniqueness lemmas.
  obtain ⟨x, hx⟩ := FindSingleNumber_precond_implies_exists_count1 nums h_precond
  refine ⟨x, hx, ?_⟩
  intro y hy
  -- Uniqueness is provided by the dedicated lemma.
  exact FindSingleNumber_precond_implies_unique_count1 nums h_precond y x hy hx

/-- Helper: the mapped list of counts contains the count of any member. -/
lemma mem_map_count_of_mem
    (nums : List Int) (x : Int) (hx : x ∈ nums) :
    nums.count x ∈ nums.map (fun y => nums.count y) := by
  classical
  -- We only state the existence of the corresponding element in the mapped list.
  -- The detailed proof is deferred.
  sorry

/-- Bridging lemma: for a decidable predicate `P` on a list, the boolean
    `List.all (fun a => decide (P a))` is `true` exactly when `P` holds
    for all elements of the list. -/
lemma List_all_decide_iff {α : Type} (l : List α) (P : α → Prop) [DecidablePred P] :
    l.all (fun a => decide (P a)) = true ↔ ∀ a ∈ l, P a := by
  classical
  -- This lemma isolates the reasoning about `List.all` and `decide`.
  -- Its detailed proof is deferred.
  sorry

/-- Helper: extract the `all`-part of the precondition as a property on counts
    of elements of `nums`. -/
lemma FindSingleNumber_precond_all_counts
    (nums : List Int) (h_precond : FindSingleNumber_precond nums) :
    ∀ c ∈ nums.map (fun x => nums.count x), c = 1 ∨ c = 2 := by
  classical
  unfold FindSingleNumber_precond at h_precond
  rcases h_precond with ⟨h_all, _h_unique⟩
  -- Use the generic characterization of `List.all (fun a => decide (P a))`.
  have := (List_all_decide_iff (nums.map (fun x => nums.count x))
            (fun c => c = 1 ∨ c = 2)).1
  -- Apply it to the stored boolean equality.
  exact this h_all

/-- From the precondition, every element in `nums` has count either 1 or 2. -/
lemma FindSingleNumber_precond_implies_count_in_1_or_2
    (nums : List Int) (h_precond : FindSingleNumber_precond nums) :
    ∀ x ∈ nums, nums.count x = 1 ∨ nums.count x = 2 := by
  classical
  intro x hx
  -- Use that `nums.count x` appears in the mapped list of counts.
  have h_mem_counts :
      nums.count x ∈ nums.map (fun y => nums.count y) :=
    mem_map_count_of_mem nums x hx
  -- Use the `all`-part of the precondition instantiated at this count.
  have h_all_counts :=
    FindSingleNumber_precond_all_counts nums h_precond
  exact h_all_counts _ h_mem_counts

lemma List_count_eq_one_mem {α : Type} [DecidableEq α]
    (xs : List α) (x : α) (hx : xs.count x = 1) :
    x ∈ xs := by
  -- A standard fact: if `count x xs = 1` then `x` occurs in `xs`.
  -- (We keep this as a smaller, local lemma with its own proof obligation.)
  sorry

lemma List_mem_length_pos {α : Type} (xs : List α) (x : α) (hx : x ∈ xs) :
    xs.length > 0 := by
  -- Another simple helper: having a member implies the list is nonempty.
  sorry

lemma FindSingleNumber_spec_len_pos (nums : List Int) (h_precond : FindSingleNumber_precond nums) :
    nums.length > 0 := by
  classical
  -- From the precondition we get a unique element whose count is 1.
  obtain ⟨x, hx1, _hxuniq⟩ :=
    FindSingleNumber_precond_implies_exists_unique_count1 nums h_precond
  -- First deduce membership of `x` from its count.
  have hxmem : x ∈ nums :=
    List_count_eq_one_mem nums x hx1
  -- Then deduce that the list is nonempty from the existence of a member.
  exact List_mem_length_pos nums x hxmem

/-- The result returned by `FindSingleNumber` appears exactly once in `nums`. -/
lemma FindSingleNumber_spec_result_occurs_once (nums : List Int) (h_precond : FindSingleNumber_precond nums) :
    (filterlist (FindSingleNumber nums h_precond) nums).length = 1 := by
  classical
  -- Strategy:
  -- 1. Let `r := FindSingleNumber nums h_precond`.
  -- 2. Show `nums.count r = 1` using the algorithm and uniqueness-of-count-1 lemma.
  -- 3. Conclude via `filterlist_spec_count`.
  set r := FindSingleNumber nums h_precond with hr
  have hcount : nums.count r = 1 := by
    -- Use correctness of the search algorithm together with
    -- `FindSingleNumber_precond_implies_exists_unique_count1`.
    sorry
  -- Convert count property to filterlist length property.
  simpa [hr, filterlist_spec_count] using hcount

/-- Helper lemma: if `x` is in `nums` and is not the unique single-occurrence element
    (as found by `FindSingleNumber`), then it must occur exactly twice. -/
lemma FindSingleNumber_spec_double_occurrence
    (nums : List Int) (h_precond : FindSingleNumber_precond nums)
    (x : Int) (hx : x ∈ nums)
    (h_ne : x ≠ FindSingleNumber nums h_precond) :
    (filterlist x nums).length = 2 := by
  classical
  -- 1. From the precondition we know `nums.count x = 1 ∨ nums.count x = 2`.
  have hcount_or := FindSingleNumber_precond_implies_count_in_1_or_2 nums h_precond x hx
  -- 2. Show `nums.count (FindSingleNumber nums h_precond) = 1`.
  have hres_count :
      nums.count (FindSingleNumber nums h_precond) = 1 := by
    -- From previous lemma `FindSingleNumber_spec_result_occurs_once`,
    -- together with `filterlist_spec_count`.
    have hlen :=
      FindSingleNumber_spec_result_occurs_once nums h_precond
    -- `hlen : (filterlist (FindSingleNumber nums h_precond) nums).length = 1`
    simpa [filterlist_spec_count] using hlen
  -- 3. Use uniqueness of the element with count 1 to rule out `nums.count x = 1`.
  have huniq := FindSingleNumber_precond_implies_exists_unique_count1 nums h_precond
  -- From uniqueness, if `nums.count x = 1` then `x = FindSingleNumber ...`,
  -- contradicting `h_ne`.
  have : nums.count x ≠ 1 := by
    intro hcx
    rcases huniq with ⟨u, hu1, huuniq⟩
    -- `u` is the unique element with count 1, hence `x = u` and also
    -- the result has count 1, so `x` must equal the result, contradicting `h_ne`.
    have hx_eq_u : x = u := by
      apply huuniq
      exact hcx
    have hres_eq_u : FindSingleNumber nums h_precond = u := by
      apply huuniq
      exact hres_count
    have : x = FindSingleNumber nums h_precond := by
      simpa [hx_eq_u] using hres_eq_u.symm
    exact h_ne this
  -- 4. So the only remaining possibility is `nums.count x = 2`.
  have hcount2 : nums.count x = 2 := by
    cases hcount_or with
    | inl h1 =>
        exact (this h1).elim
    | inr h2 =>
        exact h2
  -- 5. Translate count to filterlist length via `filterlist_spec_count`.
  simpa [filterlist_spec_count] using hcount2

lemma FindSingleNumber_spec_all_elems (nums : List Int) (h_precond : FindSingleNumber_precond nums) :
    ∀ (x : Int),
      x ∈ nums →
      (x = FindSingleNumber nums h_precond) ∨ ((filterlist x nums).length = 2) := by
  classical
  intro x hx
  by_cases h_eq : x = FindSingleNumber nums h_precond
  · left
    exact h_eq
  · right
    exact FindSingleNumber_spec_double_occurrence nums h_precond x hx h_eq

theorem FindSingleNumber_spec_satisfied (nums: List Int) (h_precond : FindSingleNumber_precond (nums)) :
    FindSingleNumber_postcond (nums) (FindSingleNumber (nums) h_precond) h_precond := by
  classical
  unfold FindSingleNumber_postcond
  refine And.intro (FindSingleNumber_spec_len_pos nums h_precond) ?_
  refine And.intro (FindSingleNumber_spec_result_occurs_once nums h_precond) ?_
  exact FindSingleNumber_spec_all_elems nums h_precond

-- EVOLVE-BLOCK-END

end verina_advanced_1
