import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from transformers import OPTForCausalLM, AutoModelForCausalLM, OPTModel
    
class BaseHFModel:
    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        model = super().from_pretrained(model_name_or_path, **kwargs)
        
        # print(model)
        
        model.set_linear(model.config.linear_dim)
        ##
        return model
    
    def set_linear(self, linear_dim):
        
        if 'opt-350m' in self.config._name_or_path:
            hidden_size = self.config.word_embed_proj_dim
        else:
            hidden_size = self.config.hidden_size
        
        self.score = nn.Linear(hidden_size, linear_dim, bias=False)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):

        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        
        pooled_output = outputs.last_hidden_state
        pooled_logits = self.score(pooled_output)
        
        if labels is None:
            return pooled_logits[:, -1]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(pooled_logits.view(-1, pooled_logits.size(-1)), labels.view(-1))
            
        return loss

class OPTRecModel(BaseHFModel, OPTModel):
    def __init__(self, config):
        super().__init__(config)
        
        
if __name__ == "__main__":
    import os
    from transformers import AutoConfig, AutoTokenizer
    from parameters import get_args
    from models_utils import print_trainable_parameters
    args = get_args()
    
    kwargs = {"cache_dir": args.model_cache_dir, "local_files_only": os.path.exists(args.model_cache_dir)}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **kwargs)
    print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('.')))
    
    config = AutoConfig.from_pretrained(args.model_name_or_path, **kwargs)
    config.linear_dim = 1234
    config.vocab_size = 12345
    
    # model = OPTRecModel.from_pretrained(args.model_name_or_path, config=config, **kwargs)
    # print(model)
    # model.set_linear(123)
    # print(model)
    def print_trainable_parameters(args, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                print(_, param.numel())
                trainable_params += param.numel()
        if args.bits == 4: 
            trainable_params /= 2

        print(
            f"trainable params: {trainable_params} || "
            f"all params: {all_param} || "
            f"trainable: {100 * trainable_params / all_param}"
        )
    model = OPTRecModel(config)
    model.set_linear(config.linear_dim)
    print(model)
    model.decoder.embed_tokens.weight.requires_grad = False
    for _ in model.decoder.parameters():
        print(_.requires_grad)
    #     _.requires_grad = False
    print_trainable_parameters(args, model)

