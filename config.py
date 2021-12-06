class Config():
    def __init__(self, vocab_size=30000, max_position_embeddings=512, columns=1, hidden_size=768, layer_norm_eps=1e-12, hidden_dropout_prob=0.1):
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.columns = columns

    def getDict(self):
        return {"vocab_size": self.vocab_size, "max_position_embeddings": self.max_position_embeddings, "hidden_size": self.hidden_size, "layer_norm_eps": self.layer_norm_eps, "hidden_dropout_prob": self.hidden_dropout_prob, "columns": self.columns}

    def fromDict(self, dict):
        self.hidden_size = dict["hidden_size"]
        self.max_position_embeddings = dict["max_position_embeddings"]
        self.vocab_size = dict["vocab_size"]
        self.layer_norm_eps = dict["layer_norm_eps"]
        self.hidden_dropout_prob = dict["hidden_dropout_prob"]
        self.columns = dict["columns"]


class LayerConfig():
    def __init__(self, hidden_size=768, num_attention_heads=12, intermediate_size=3072, layer_norm_eps=1e-12, hidden_dropout_prob=0.1, attention_type="selfAttention"):

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_type = attention_type
        # gmlp
        self.tinyAtt = True
        self.causal = False
        self.attn_dim = 64

    def getDict(self):
        return {"hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "layer_norm_eps": self.layer_norm_eps,
                "hidden_dropout_prob": self.hidden_dropout_prob,
                "attention_type":self.attention_type}

    def fromDict(self, dict):
        self.hidden_size = dict["hidden_size"]
        self.num_attention_heads = dict["num_attention_heads"]
        self.intermediate_size = dict["intermediate_size"]
        self.layer_norm_eps = dict["layer_norm_eps"]
        self.hidden_dropout_prob = dict["hidden_dropout_prob"]
        self.attention_type = dict["attention_type"]
