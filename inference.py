import torch
import torch.nn.functional as F
import pandas as pd
import os

def predict(model, test_loader, device,path):
    model.eval()
    target = []
    for batch_num, (captions, input_id, attention_masks) in enumerate(test_loader):


        input_ids, attention_masks = input_id.to(device), attention_masks.to(device)
        output_dictionary = model(input_ids,
                                  token_type_ids=None,
                                  attention_mask=attention_masks,
                                  return_dict=True)

        predictions = F.softmax(output_dictionary['logits'], dim=1)

        _, top1_pred_labels = torch.max(predictions ,1)
        top1_pred_labels = top1_pred_labels.item()
        target.append(top1_pred_labels)


    make_csv(target,path)


def make_csv(target,path):
    test = pd.read_csv(os.path.join(path,'test.csv'))
    my_submission = pd.DataFrame({'id': test.id, 'target': target})
    my_submission.to_csv(os.path.join(path,'submission.csv'), index=False)