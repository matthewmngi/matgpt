import torch
from torch import nn
from torch.nn import functional as F

# ------------------------

with open(f"data/dialogues_train.txt", "r", encoding="utf-8") as f:
    text = f.read()
    
# ------------------------
    
@torch.no_grad()
def estimate_loss():
    out = {}
    net.eval()
    
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            X, y = get_batch(split)
            logits, loss = net(X, y)
            losses[k] = loss.item()
            
        out[split] = losses.mean()
    
    net.train()
    return out

# ------------------------

text = text.replace("__eou__ ", "")

chars = sorted(list(set(text)))
vocab_size = len(chars)

# ------------------------

s_i = { ch : i for i, ch in enumerate(chars) }
i_s = { i : ch for i, ch in enumerate(chars) }

encode = lambda s: [s_i[c] for c in s]
decode = lambda l: "".join([i_s[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.8*len(data))

# ------------------------

dtr = data[:n]
dval = data[n:]

torch.manual_seed(1337)
batch_size = 4
block_size = 8 # max context length

def get_batch(split):
    data = dtr if split == "train" else dval
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    
    return x, y

# ------------------------

torch.manual_seed(1337)

n_embd = 32

class Bigram(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embd = nn.Embedding(vocab_size, n_embd)
        self.position_embd = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embd(idx) # (B, T, C)
        pos_emb = self.position_embd(torch.arange(T)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            
            targets = targets.view(-1)
            
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # (B, C)
            
            probs = F.softmax(logits, dim=-1) # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx
    
net = Bigram()
Xb, yb = get_batch("train")
out, loss = net(Xb, yb)

# ------------------------

optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)

batch_size = 32
max_iter = 10000

for step in range(max_iter):
    Xb, yb = get_batch("train")
    
    logits, loss = net(Xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    
    loss.backward()
    optimizer.step()
    
    if step % 500 == 0:
        print(f'{step} / {max_iter}: {loss.item()}')
        
print("Final loss:", loss)