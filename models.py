import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from transformers import BertModel

huggingface_model='bert-base-multilingual-uncased'

class BERT_Arch(nn.Module):

    def __init__(self, bert,num_classes):

        super(BERT_Arch, self).__init__()
        self.bert = bert  
        self.num_classes=num_classes     
        # dropout layer
        self.dropout = nn.Dropout(0.3)      
        # dense layer (Output layer)
        self.fc1 = nn.Linear(768,num_classes)
        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

        #pass the inputs to the model  
        out = self.bert(sent_id, attention_mask=mask) 
        x=torch.mean(out.last_hidden_state, dim=1).squeeze(dim=1)
        x = self.dropout(x)
        # output layer
        x = self.fc1(x)    
        # apply softmax activation
        x = self.softmax(x)
        return x
    
    @torch.inference_mode()
    def predict(self, batch):
        self.eval()
        ids, masks = batch["ids"], batch["masks"]
        z = self(ids,masks)
        y_pred = torch.argmax(z, dim=1).cpu().numpy()
        return y_pred
    
    @torch.inference_mode()
    def predict_proba(self, batch):
        self.eval()
        ids, masks = batch["ids"], batch["masks"]
        z = self(ids, masks)
        y_probs = z
        #y_probs = torch.argmax(z, dim=1).cpu().numpy()
        print(y_probs)
        #y_probs = F.softmax(z, dim=1).cpu().numpy()
        return y_probs
    
    def save(self, dp):
        with open(Path(dp, "args.json"), "w") as fp:
            contents = {
                #"dropout_p": self.dropout_p,
                #"embedding_dim": self.embedding_dim,
                "num_classes": self.num_classes,
            }
            json.dump(contents, fp, indent=4, sort_keys=False)
        torch.save(self.state_dict(), os.path.join(dp, "model.pt"))
        
    @classmethod
    def load(cls, args_fp, state_dict_fp):
        with open(args_fp, "r") as fp:
            kwargs = json.load(fp=fp)
        llm = BertModel.from_pretrained(huggingface_model, return_dict=True)
        model = cls(bert=llm, **kwargs)
        model.load_state_dict(torch.load(state_dict_fp, map_location=torch.device("cpu")))
        #print(model)
        return model