# Training Data

Place your training corpus here as `input.txt`.

## Recommended datasets

**Tiny Shakespeare** (~1MB, great for quick tests):
```
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

**Other options:**
- Any plain-text `.txt` file works — public domain books, Wikipedia dumps, etc.
- Aim for at least 1MB of text for meaningful training with a 2M-param model.

## Expected loss

With a character-level vocabulary of ~65 chars, random-weight initial loss is:
```
ln(65) ≈ 4.17
```
A well-trained model on Tiny Shakespeare should reach ~1.3–1.5 nats.
