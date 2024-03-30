import json
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from torch.utils.tensorboard import SummaryWriter

TRAIN_DATA_PATH = "./data/training.json"
VALIDATION_DATA_PATH = "./data/validation.json"
TEST_DATA_PATH = "./data/test.json"
LABEL_PATH = "./data/label.json"

NUM_LABELS = 753

insurance_options = ['Self Pay', 'Medicaid', 'Medicare', 'Private', 'Government']
religion_options = ['CHRISTIAN SCIENTIST', 'PROTESTANT QUAKER', 'NOT SPECIFIED', 'OTHER', 'EPISCOPALIAN', 'ROMANIAN EAST. ORTH', 'UNOBTAINABLE', 'GREEK ORTHODOX', 'JEWISH', 'CATHOLIC']
marital_status_options = ['WIDOWED', 'MARRIED', 'DIVORCED', 'SEPARATED', 'SINGLE']
ethnicity_options = ['UNKNOWN/NOT SPECIFIED', 'HISPANIC/LATINO - GUATEMALAN', 'MULTI RACE ETHNICITY', 'UNABLE TO OBTAIN', 'BLACK/CAPE VERDEAN', 'HISPANIC/LATINO - PUERTO RICAN', 'BLACK/HAITIAN', 'PATIENT DECLINED TO ANSWER', 'OTHER', 'ASIAN', 'HISPANIC OR LATINO', 'BLACK/AFRICAN AMERICAN', 'WHITE', 'WHITE - BRAZILIAN']


class Model(torch.nn.Module):
    def __init__(self, num_labels):
        super(Model, self).__init__()
        # load the ClinicalBERT model
        self.bert = BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        # freeze the model
        for param in self.bert.parameters():
            param.requires_grad = False
        # add a linear layer
        self.fc = torch.nn.Linear(788 + num_labels, 1000)
        self.fc2 = torch.nn.Linear(1000, num_labels)

        # Create embeddings for additional features
        self.insurance_embeddings = torch.nn.Embedding(len(insurance_options), 5)
        self.religion_embeddings = torch.nn.Embedding(len(religion_options), 5)
        self.marital_status_embeddings = torch.nn.Embedding(len(marital_status_options), 5)
        self.ethnicity_embeddings = torch.nn.Embedding(len(ethnicity_options), 5)


    def forward(self, x):
        bert_output = self.bert(x['input_ids'].squeeze(), x['attention_mask'].squeeze())
        
        bert_output = bert_output.pooler_output
        
        insurance = self.insurance_embeddings(x['insurance'])
        religion = self.religion_embeddings(x['religion'])
        marital_status = self.marital_status_embeddings(x['marital_status'])
        ethnicity = self.ethnicity_embeddings(x['ethnicity'])
        prev_labels = x['prev_labels']



        x = torch.cat([bert_output, insurance, religion, marital_status, ethnicity, prev_labels], dim=1)
        
        # get the final output of the bert model
        x = self.fc(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]

        text_inputs = self.tokenizer(
            record['note_text'],
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding='max_length'
        )
        inputs = {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'insurance': torch.tensor(self.get_option_index(record['insurance'], insurance_options)),
            'religion': torch.tensor(self.get_option_index(record['religion'], religion_options)),
            'marital_status': torch.tensor(self.get_option_index(record['marital_status'], marital_status_options)),
            'ethnicity': torch.tensor(self.get_option_index(record['ethnicity'], ethnicity_options)),
        }

        # Previous lab events
        prev_labels = [0] * NUM_LABELS
        for label in record['prev_labels']:
            prev_labels[label] = 1
        prev_labels = torch.tensor(prev_labels)
        inputs['prev_labels'] = prev_labels

        labels = [0] * NUM_LABELS
        for label in record['labels']:
            labels[label] = 1
        labels = torch.tensor(labels)

        return inputs, labels

    def get_option_index(self, value, options):
        if value in options:
            return options.index(value)
        return 0


if __name__ == "__main__":

    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 
    device = torch.device(dev)
    print('Device:', device)

    # read the labels
    label_map = json.load(open(LABEL_PATH))
    num_labels = len(label_map)
    
    # load the datasets
    training_dataset = json.load(open(TRAIN_DATA_PATH))   
    training_dataset = Dataset(training_dataset)
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=8, shuffle=True)

    validation_dataset = json.load(open(VALIDATION_DATA_PATH))
    validation_dataset = Dataset(validation_dataset)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=8, shuffle=False)

    testing_dataset = json.load(open(TEST_DATA_PATH))
    testing_dataset = Dataset(testing_dataset)
    testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=8, shuffle=False)

    writer = SummaryWriter()

    model = Model(num_labels)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(10):
        model.train()
        for batch_idx, (inputs, labels) in tqdm(enumerate(training_loader)):
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            
            loss = criterion(output, labels.float())
            loss.backward()
            optimizer.step()

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item()}")

                writer.add_scalar('Loss/train', loss.item(), epoch * len(training_loader) + batch_idx)

                # calculate the mean error 
                mean_error = torch.mean(torch.abs(torch.sigmoid(output) - labels.float()))
                print(f"Mean Error: {mean_error.item()}")

        print(f"Epoch {epoch} Loss: {loss.item()}")

        # save the model
        torch.save(model.state_dict(), 'model.pth')

        # evaluate the model
        model.eval()
        total_loss = 0
        total_mean_error = 0
        correct_predictions = 0
        for batch_idx, (inputs, labels) in enumerate(validation_loader):
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            labels = labels.to(device)
            
            output = model(inputs)
            loss = criterion(output, labels.float())
            total_loss += loss.item()

            mean_error = torch.mean(torch.abs(torch.sigmoid(output) - labels.float()))
            total_mean_error += mean_error.item()

            predictions = torch.sigmoid(output) > 0.5
            predictions = predictions.float()
            correct_predictions += torch.sum(predictions == labels).item()

        print(f"Validation Loss: {total_loss / len(validation_loader)}")
        print(f"Validation Mean Error: {total_mean_error / len(validation_loader)}")
        print(f"Validation Accuracy: {correct_predictions / (len(validation_dataset) * num_labels)}")

        writer.add_scalar('Loss/validation', total_loss / len(validation_loader), epoch)
        writer.add_scalar('Mean Error/validation', total_mean_error / len(validation_loader), epoch)
        writer.add_scalar('Accuracy/validation', correct_predictions / (len(validation_dataset) * num_labels), epoch)


    # evaluate the model on the test set
    model.eval()
    total_loss = 0
    total_mean_error = 0
    correct_predictions = 0
    for batch_idx, (inputs, labels) in enumerate(testing_loader):
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        labels = labels.to(device)
        
        output = model(inputs)
        loss = criterion(output, labels.float())
        total_loss += loss.item()

        mean_error = torch.mean(torch.abs(torch.sigmoid(output) - labels.float()))
        total_mean_error += mean_error.item()

        predictions = torch.sigmoid(output) > 0.5
        predictions = predictions.float()
        correct_predictions += torch.sum(predictions == labels).item()

    print(f"Test Loss: {total_loss / len(testing_loader)}")
    print(f"Test Mean Error: {total_mean_error / len(testing_loader)}")
    print(f"Test Accuracy: {correct_predictions / (len(testing_dataset) * num_labels)}")
    writer.add_scalar('Loss/test', total_loss / len(testing_loader), epoch)
    writer.add_scalar('Mean Error/test', total_mean_error / len(testing_loader), epoch)
    writer.add_scalar('Accuracy/test', correct_predictions / (len(testing_dataset) * num_labels), epoch)

    writer.close()



