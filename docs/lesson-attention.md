# Lesson: Self-Attention from First Principles

## Chapter 0: Foundations

### 0.1 — Vectors

A vector is just a list of numbers:

```
a = [3.0, 1.0, 4.0]
```

That's it. No magic. In our model, each token gets represented as a vector
of 128 numbers. Why 128? It's a design choice — more numbers = the model can
store richer information about each token, but uses more memory.

Think of a vector as a **description** using numbers instead of words.
A color could be described as `[red=0.9, green=0.1, blue=0.0]`. A token's
embedding is the same idea, just with 128 abstract features instead of 3
obvious ones.

### 0.2 — Matrices

A matrix is a grid of numbers — rows and columns:

```
    col0  col1  col2
   ┌─────────────────┐
row0│ 1.0   2.0   3.0 │
row1│ 4.0   5.0   6.0 │
   └─────────────────┘
```

Shape is written as `[rows, columns]`, so this is `[2, 3]`.

A matrix can also be thought of as a **stack of vectors** — each row is one
vector. Our input `[T, D]` is T vectors (one per token) stacked on top of
each other.

### 0.3 — Matrix Multiplication (Matmul)

You implemented this already, but let's make sure the *meaning* is clear,
not just the code.

Given A `[M, K]` and B `[K, N]`, the result is `[M, N]`.

**The rule:** element `[i, j]` in the result = take row `i` from A, take
column `j` from B, multiply them element by element, and sum up.

Concrete example:

```
A = [1, 2]     B = [5, 6]
    [3, 4]         [7, 8]

Result[0,0] = 1*5 + 2*7 = 19
Result[0,1] = 1*6 + 2*8 = 22
Result[1,0] = 3*5 + 4*7 = 43
Result[1,1] = 3*6 + 4*8 = 50

Result = [19, 22]
         [43, 50]
```

**Why it matters for ML:** matmul is how we apply **learned transformations**.
When you multiply input `[T, D]` by weights `[D, out_dim]`, you're
transforming each token from a D-dimensional description into an
out_dim-dimensional description. The weight values determine *how* the
features get mixed and remapped.

**Shape intuition:** the inner dimensions must match (the K in `[M,K]` and
`[K,N]`). The outer dimensions become the result shape (`[M, N]`). Think of
it as: "M things in, each described by K features, transformed into N new
features."

```
[T, D] × [D, out] = [T, out]
 ↑         ↑   ↑      ↑   ↑
 T tokens  must match  T tokens, new description size
```

### 0.4 — Transpose

Transposing a matrix flips rows and columns:

```
Original [2, 3]:        Transposed [3, 2]:
[1, 2, 3]               [1, 4]
[4, 5, 6]               [2, 5]
                         [3, 6]
```

Row 0 becomes column 0. Row 1 becomes column 1.

**Why we need this in attention:** matmul takes row-from-A and column-from-B.
But sometimes we want to compute dot products between rows of two matrices.
If we transpose B first, then B's rows become columns, and matmul gives us
every row-vs-row dot product.

```
A has 3 rows, B has 3 rows, each of length 4.

A.matmul(&B.transpose_2d())
  [3, 4] × [4, 3] = [3, 3]

Result[i][j] = dot product of A's row i with B's row j
```

This is exactly how attention computes "how much does token i care about
token j."

### 0.5 — Softmax

Softmax takes any list of numbers and turns them into **probabilities** — all
positive, summing to 1.0.

```
Input:  [2.0, 1.0, 0.1]

Step 1 — exponentiate each:   [e^2.0, e^1.0, e^0.1] = [7.39, 2.72, 1.11]
Step 2 — sum them up:         7.39 + 2.72 + 1.11 = 11.22
Step 3 — divide each by sum:  [0.659, 0.242, 0.099]

Output: [0.659, 0.242, 0.099]    (sums to 1.0)
```

**Key properties:**
- Bigger input → bigger output (2.0 became 0.659, the largest)
- All outputs are positive (because `e^x` is always positive)
- They sum to 1.0 (because we divided by the total)
- **`e^(-infinity) = 0`** — this is how the causal mask works. Set a score
  to `-inf` and softmax turns it into exactly 0.0. The token is invisible.

**Row-wise softmax** (which you implemented as `softmax_rows`) does this
independently for each row of a matrix. Each row becomes its own probability
distribution.

```
Before softmax_rows:          After:
[2.0,  1.0, -inf]            [0.731, 0.269, 0.000]  ← sums to 1.0
[1.0,  1.0,  1.0]            [0.333, 0.333, 0.333]  ← sums to 1.0
```

In attention, each row represents one token's attention scores. After
softmax, it becomes that token's attention **weights** — how much to look
at each other token.

### 0.6 — Weighted Sum

The last building block. If you have weights `[0.7, 0.2, 0.1]` and three
vectors:

```
v0 = [1.0, 0.0]
v1 = [0.0, 1.0]
v2 = [0.5, 0.5]

weighted sum = 0.7 * v0 + 0.2 * v1 + 0.1 * v2
             = [0.7, 0.0] + [0.0, 0.2] + [0.05, 0.05]
             = [0.75, 0.25]
```

The result is mostly v0 (weight 0.7) with a bit of v1 and v2 mixed in.

**This is what `weights.matmul(&V)` does.** The attention weights matrix
`[T, T]` times the Value matrix `[T, head_dim]` produces a weighted sum
of Value vectors for each token. Token i's output is a blend of all Value
vectors, weighted by how much token i attends to each.

---

## Chapter 1: The Problem Attention Solves

### 1.1 — Tokens as Vectors

When we process text, each token (character, in our case) gets turned into a
vector of numbers called an **embedding**. In our model, each token becomes a
vector of 128 numbers (`embed_dim = 128`).

If our input is "hello" (5 tokens), we have a matrix shaped `[5, 128]`:

```
         128 numbers ──────────►
       ┌───────────────────────┐
"h" →  │ 0.12  0.45  ... 0.03 │  ← token 0's embedding
"e" →  │ 0.87  0.11  ... 0.56 │  ← token 1's embedding
"l" →  │ 0.33  0.92  ... 0.41 │  ← token 2's embedding
"l" →  │ 0.33  0.92  ... 0.41 │  ← token 3's embedding (same char, same embedding)
"o" →  │ 0.71  0.08  ... 0.29 │  ← token 4's embedding
       └───────────────────────┘
```

This is your input `x` with shape `[T, D]` where `T=5` and `D=128`.

### 1.2 — Why We Need Context

Right now, each token's vector is independent. "h" has the same vector whether
it appears in "hello" or "hurt". But meaning depends on context — the "l" in
"hello" is different from the "l" in "evil".

**Attention is the mechanism that lets each token look at other tokens and
update its own vector based on what it sees.**

After attention, each token's vector has been enriched with information from
the tokens around it. The "l" in "hello" will have absorbed information from
"h" and "e" before it.

### 1.3 — Why Only Look Backward?

We're building a **language model** — it predicts the next character. When
predicting what comes after "hel", the model should only see "h", "e", "l".
If it could see "l" and "o" (future tokens), it would be cheating.

So each token can only attend to itself and tokens **before** it. This is
called **causal** attention.

---

## Chapter 2: The Dot Product as Similarity

### 2.1 — Comparing Two Vectors

Before we get to Q, K, V, you need one concept: the **dot product** measures
how similar two vectors are.

```
a = [1, 0, 0]
b = [1, 0, 0]     a · b = 1*1 + 0*0 + 0*0 = 1    (identical → high)

a = [1, 0, 0]
b = [0, 1, 0]     a · b = 1*0 + 0*1 + 0*0 = 0    (unrelated → zero)

a = [1, 0, 0]
b = [-1, 0, 0]    a · b = 1*-1 + 0*0 + 0*0 = -1   (opposite → negative)
```

High dot product = vectors point in the same direction = "similar."

### 2.2 — Matmul IS Many Dot Products

When you do `A.matmul(&B)` where A is `[M, K]` and B is `[K, N]`, element
`[i, j]` in the result is the dot product of row `i` of A with column `j` of B.

So `A.matmul(&B.transpose_2d())` computes dot products between every pair of
rows from A and B. If A and B are both `[T, D]`:

```
result = A.matmul(&B.transpose_2d())    // [T, D] × [D, T] = [T, T]

result[i][j] = dot product of row i of A with row j of B
             = "how similar is token i to token j"
```

**This is the core of attention.** Everything else is setup and cleanup.

---

## Chapter 3: Query, Key, Value

### 3.1 — Why Three Separate Roles?

Imagine you're at a conference. You have a **question** you want answered
(your Query). Every other person has a **name tag** describing their expertise
(their Key) and a **document** they can share (their Value).

You compare your question to each name tag (Query vs Key). The best matches
get the highest scores. Then you read a weighted mix of everyone's documents
(weighted sum of Values).

Using the same vector for all three roles would be limiting — what you're
searching for (Q), what you're known for (K), and what you share (V) should
be different.

### 3.2 — Creating Q, K, V

Each is created by a simple linear transformation (matmul + bias) of the input.
We could use three separate weight matrices, but it's more efficient to use
one big one and split the result:

```
input:  [T, D]          (5 tokens, 128 features each)
w_qkv:  [D, 3*D]        (128 → 384)
b_qkv:  [3*D]           (384 biases)

qkv = input.matmul(&w_qkv).add_bias(&b_qkv)    // [T, 3*D] = [5, 384]
```

Then split `qkv` into three equal pieces along the columns:

```
qkv:  [T, 3*D]
       ├── columns 0..D ──┤── columns D..2D ──┤── columns 2D..3D ──┤
       │        Q          │        K          │        V           │
       │     [T, D]        │     [T, D]        │     [T, D]         │
```

Each token still has T rows. We just split the 384 columns into three groups
of 128.

### 3.3 — How to Split in Flat Data

You already know this from `add_bias`. For each row, you grab a different
range of columns:

```
For a [T, 3D] tensor, to extract Q (columns 0..D):
  For each row r:
    start = r * (3*D) + 0
    end   = r * (3*D) + D
    grab data[start..end]

To extract K (columns D..2D):
  For each row r:
    start = r * (3*D) + D
    end   = r * (3*D) + 2*D

To extract V (columns 2D..3D):
  For each row r:
    start = r * (3*D) + 2*D
    end   = r * (3*D) + 3*D
```

Result: three tensors, each `[T, D]`.

---

## Chapter 4: Multi-Head Attention

### 4.1 — Why Multiple Heads?

One attention pattern might learn "look at the previous word." Another might
learn "look at the subject of the sentence." If we only have one attention
mechanism, it has to do everything at once.

**Multi-head attention** runs several independent attention mechanisms in
parallel, each looking at a different slice of the features.

### 4.2 — Splitting Into Heads

With `embed_dim=128` and `num_heads=2`, each head works with `head_dim=64`.

We take Q `[T, 128]` and split it into:
- Head 0's Q: columns 0..63    → `[T, 64]`
- Head 1's Q: columns 64..127  → `[T, 64]`

Same for K and V. Now each head has its own Q, K, V, each `[T, 64]`.

### 4.3 — Per-Head Attention

Each head independently runs the same steps:

```
For head h:
    Q_h = [T, head_dim]     (this head's queries)
    K_h = [T, head_dim]     (this head's keys)
    V_h = [T, head_dim]     (this head's values)

    scores = Q_h.matmul(&K_h.transpose_2d())   // [T, T]
    scaled  = scores.scale(1.0 / sqrt(head_dim))
    masked  = scaled.add(&causal_mask)
    weights = masked.softmax_rows()             // [T, T], each row sums to 1
    out_h   = weights.matmul(&V_h)              // [T, head_dim]
```

### 4.4 — Concatenate Heads

After all heads are done, stitch their outputs side by side:

```
head 0 output: [T, 64]
head 1 output: [T, 64]

concatenated:  [T, 128]   (back to [T, D])
```

This is the reverse of splitting — for each row, you lay head 0's 64 values
followed by head 1's 64 values.

---

## Chapter 5: Putting It All Together

### 5.1 — The Complete Flow

```
input x: [T, D]
    │
    ▼
(1) qkv = x.matmul(&w_qkv).add_bias(&b_qkv)         [T, 3D]
    │
    ▼
(2) Split into Q, K, V                                 three [T, D]
    │
    ▼
(3) For each head h = 0..num_heads:
    │  Extract Q_h, K_h, V_h                            [T, head_dim]
    │  scores = Q_h @ K_h^T                              [T, T]
    │  scores = scores / sqrt(head_dim)
    │  scores = scores + causal_mask
    │  weights = softmax_rows(scores)                    [T, T]
    │  out_h = weights @ V_h                             [T, head_dim]
    │
    ▼
(4) Concatenate all out_h                               [T, D]
    │
    ▼
(5) output = concat.matmul(&w_proj).add_bias(&b_proj)  [T, D]
```

### 5.2 — Why the Final Projection?

The concatenated heads are just stacked side by side — head 0's features and
head 1's features don't interact. The final projection `w_proj` is a `[D, D]`
matrix that mixes them together, allowing the model to combine insights from
different heads.

### 5.3 — Shape Summary

| Step                  | Shape          | Example (T=5, D=128, H=2) |
|-----------------------|----------------|---------------------------|
| Input                 | [T, D]         | [5, 128]                  |
| After QKV projection  | [T, 3D]        | [5, 384]                  |
| Q, K, V each          | [T, D]         | [5, 128]                  |
| Per-head Q, K, V      | [T, head_dim]  | [5, 64]                   |
| Attention scores      | [T, T]         | [5, 5]                    |
| Attention weights     | [T, T]         | [5, 5]                    |
| Per-head output       | [T, head_dim]  | [5, 64]                   |
| Concatenated          | [T, D]         | [5, 128]                  |
| Final output          | [T, D]         | [5, 128]                  |

Input shape = Output shape. Attention transforms the content, not the shape.

---

## Chapter 6: Exercises

Before writing the code, make sure you can answer these:

1. If `embed_dim = 128` and `num_heads = 4`, what is `head_dim`?

2. You have a `[5, 384]` tensor (T=5, 3D=384). What flat indices hold
   row 2 of K? (Hint: K is columns 128..255, and each row has 384 columns.)

3. After softmax_rows on the masked scores, what does row 0 look like?
   (Hint: token 0 can only see itself.)

4. If all attention weights in row 3 are `[0.25, 0.25, 0.25, 0.25, 0.0]`,
   what does `weights.matmul(&V)` produce for that row? (In plain English.)

5. Why is the final output the same shape as the input?

---

## Chapter 7: Answers

1. `head_dim = 128 / 4 = 32`

2. Row 2 of the full tensor starts at index `2 * 384 = 768`.
   K occupies columns 128..255, so:
   start = `768 + 128 = 896`, end = `768 + 256 = 1024`.
   Indices 896..1024.

3. Row 0 after causal mask has `-inf` everywhere except position [0,0].
   After softmax: `[1.0, 0.0, 0.0, 0.0, 0.0]`.
   Token 0 puts 100% attention on itself (it can't see anyone else).

4. An equal average of the Value vectors of tokens 0, 1, 2, and 3.
   Token 3 attends equally to the first four tokens and ignores token 4
   (weight 0.0 means token 4's value is ignored).

5. Because attention only changes *what information each token carries*,
   not how many tokens there are or how big each vector is. It's a
   content transformation, not a shape transformation.
