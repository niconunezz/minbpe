class BasicTokenizer:
    def __init__(self):
        pass

    
    def stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
    
        return counts
    
    def merge(self, ids, pair, sub):
        n_ids = []
        i = 0
        while i < len(ids):

            if (i < len(ids)-1) and (ids[i] == pair[0]) and (ids[i+1] == pair[1]):
                n_ids.append(sub)
                i += 2
                
            
            else:
                n_ids.append(ids[i])
                i += 1
        
        return n_ids
    
        
    def train(self, text, vocab_size, verbose = False):
        tokens = text.encode("utf-8")
        if verbose:
            print(f"original size: {len(tokens)}")
        ids = list(map(int, tokens))
        n_merges = vocab_size - 256

        merges = {}
        for i in range(n_merges):
            ntoken = 256 + i
            counts = self.stats(ids)
            
            mcp = max(counts, key = counts.get)

            if verbose:
                print(f" Changing {mcp} --> {ntoken} ")
                if mcp[0]<256 and mcp[1]<256:
                    print(f" This is ({chr(mcp[0])}|{chr(mcp[1])}) .")
            
            merges[mcp] = ntoken
            ids = self.merge(ids, mcp, ntoken)
            
        return ids, merges
    
    def decode(self, ids, vocab):
 
        tokens = b"".join([vocab[i] for i in ids])

        return tokens.decode("utf-8", errors = "replace")
    
    
    def encode(self, ids, merges):

        ids = ids.encode("utf-8")

        while len(ids) > 1:
            counts = self.stats(ids)

            pair = min(counts, key = lambda k: merges.get(k, float("inf")))

            if pair not in merges:
                break
                
            ids = self.merge(ids, pair, merges[pair])
        
        return ids